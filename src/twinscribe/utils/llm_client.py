"""
OpenRouter LLM Client Wrapper.

Provides a unified async client for interacting with LLMs through OpenRouter.
Supports both Anthropic (Claude) and OpenAI (GPT) models via the OpenRouter API.

Features:
- Unified interface for all supported models
- Automatic retry with exponential backoff using tenacity
- Rate limiting to respect API limits
- Token usage tracking for cost calculation
- Structured output support with JSON mode
- Streaming support for real-time responses
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from twinscribe.config.environment import get_api_key
from twinscribe.config.models import DEFAULT_MODELS, ModelConfig

logger = logging.getLogger(__name__)


# OpenRouter API configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_ENDPOINT = "/chat/completions"


class LLMClientError(Exception):
    """Base exception for LLM client errors."""

    pass


class AuthenticationError(LLMClientError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(LLMClientError):
    """Raised when rate limit is exceeded."""

    pass


class ModelNotFoundError(LLMClientError):
    """Raised when requested model is not available."""

    pass


class APIError(LLMClientError):
    """Raised for general API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TokenUsage:
    """Token usage statistics for a request.

    Attributes:
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        total_tokens: Total tokens used
        model: Model that was used
        cost_usd: Estimated cost in USD
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> TokenUsage:
        """Add another TokenUsage to this one."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model=self.model or other.model,
            cost_usd=self.cost_usd + other.cost_usd,
        )


@dataclass
class LLMResponse:
    """Response from an LLM request.

    Attributes:
        content: The text content of the response
        model: Model that generated the response
        usage: Token usage statistics
        finish_reason: Why generation stopped
        raw_response: Raw response data from API
        latency_ms: Request latency in milliseconds
    """

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        """Convert response to JSON-serializable dict."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "cost_usd": self.usage.cost_usd,
            },
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Role of the message sender (system, user, assistant)
        content: Text content of the message
    """

    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to API format."""
        return {"role": self.role, "content": self.content}


class UsageTracker:
    """Tracks cumulative token usage and costs.

    Thread-safe accumulator for token usage across multiple requests.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._usage_by_model: dict[str, TokenUsage] = {}
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.time()

    async def track(self, usage: TokenUsage) -> None:
        """Record token usage from a request.

        Args:
            usage: TokenUsage from a completed request
        """
        async with self._lock:
            self._total_requests += 1
            model = usage.model or "unknown"
            if model not in self._usage_by_model:
                self._usage_by_model[model] = TokenUsage(model=model)
            self._usage_by_model[model] = self._usage_by_model[model].add(usage)

    async def track_error(self) -> None:
        """Record an error."""
        async with self._lock:
            self._total_errors += 1

    async def get_summary(self) -> dict[str, Any]:
        """Get usage summary.

        Returns:
            Dict with usage statistics by model and totals
        """
        async with self._lock:
            total_prompt = sum(u.prompt_tokens for u in self._usage_by_model.values())
            total_completion = sum(u.completion_tokens for u in self._usage_by_model.values())
            total_cost = sum(u.cost_usd for u in self._usage_by_model.values())

            return {
                "by_model": {
                    model: {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                        "cost_usd": u.cost_usd,
                    }
                    for model, u in self._usage_by_model.items()
                },
                "totals": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_prompt + total_completion,
                    "cost_usd": total_cost,
                    "requests": self._total_requests,
                    "errors": self._total_errors,
                    "duration_seconds": time.time() - self._start_time,
                },
            }

    def reset(self) -> None:
        """Reset all tracked usage."""
        self._usage_by_model.clear()
        self._total_requests = 0
        self._total_errors = 0
        self._start_time = time.time()


class RateLimiter:
    """Simple rate limiter using token bucket algorithm.

    Limits requests per minute to avoid API rate limits.
    """

    def __init__(self, requests_per_minute: int = 60) -> None:
        self._rpm = requests_per_minute
        self._tokens = float(requests_per_minute)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request can be made."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self._rpm, self._tokens + elapsed * (self._rpm / 60.0))
            self._last_update = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) * (60.0 / self._rpm)
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class AsyncLLMClient:
    """Async client for OpenRouter LLM API.

    Provides a unified interface for calling various LLM models
    through OpenRouter with automatic retry, rate limiting,
    and usage tracking.

    Example:
        async with AsyncLLMClient() as client:
            response = await client.send_message(
                model="claude-sonnet-4-5",
                messages=[Message(role="user", content="Hello!")],
            )
            print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
        requests_per_minute: int = 60,
        http_referer: str | None = None,
        app_title: str = "TwinScribe",
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: OpenRouter API key (defaults to env var)
            base_url: Base URL for OpenRouter API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            requests_per_minute: Rate limit
            http_referer: HTTP Referer header for OpenRouter
            app_title: Application title for OpenRouter headers
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._rate_limiter = RateLimiter(requests_per_minute)
        self._usage_tracker = UsageTracker()
        self._http_referer = http_referer
        self._app_title = app_title
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> AsyncLLMClient:
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            # Get API key from environment if not provided
            api_key = self._api_key
            if api_key is None:
                api_key = get_api_key("openrouter")
            if api_key is None:
                raise AuthenticationError(
                    "OpenRouter API key not found. Set OPENROUTER_API_KEY "
                    "environment variable or pass api_key to constructor."
                )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": self._app_title,
            }
            if self._http_referer:
                headers["HTTP-Referer"] = self._http_referer

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration by name.

        Args:
            model_name: Model name or alias

        Returns:
            ModelConfig for the model

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_name in DEFAULT_MODELS:
            return DEFAULT_MODELS[model_name]

        # Check if it's a full model name (not an alias)
        for config in DEFAULT_MODELS.values():
            if config.name == model_name:
                return config

        raise ModelNotFoundError(f"Unknown model: {model_name}")

    def _calculate_cost(
        self,
        model_config: ModelConfig,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            model_config: Model configuration
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        input_cost = (prompt_tokens / 1_000_000) * model_config.cost_per_million_input
        output_cost = (completion_tokens / 1_000_000) * model_config.cost_per_million_output
        return input_cost + output_cost

    def _build_request_body(
        self,
        model_config: ModelConfig,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        stop_sequences: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the request body for the API call.

        Args:
            model_config: Model configuration
            messages: List of messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_mode: Whether to request JSON output
            stop_sequences: Stop sequences

        Returns:
            Request body dict
        """
        # Use full model name with provider prefix for OpenRouter
        model_name = model_config.name

        # OpenRouter expects provider/model format
        # Detect provider from model name pattern
        if "/" not in model_name:
            if model_name.startswith("claude") or model_name.startswith("anthropic"):
                full_model = f"anthropic/{model_name}"
            elif model_name.startswith("gpt") or model_name.startswith("openai"):
                full_model = f"openai/{model_name}"
            elif model_name.startswith("o1") or model_name.startswith("o3"):
                # OpenAI's o1/o3 models
                full_model = f"openai/{model_name}"
            else:
                # For other models, try to use as-is (may already have provider prefix)
                full_model = model_name
        else:
            full_model = model_name

        body: dict[str, Any] = {
            "model": full_model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens or model_config.max_tokens,
        }

        # Set temperature (default to model config value)
        if temperature is not None:
            body["temperature"] = temperature
        else:
            body["temperature"] = model_config.temperature

        # JSON mode
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        # Stop sequences
        if stop_sequences:
            body["stop"] = stop_sequences

        return body

    async def send_message(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        stop_sequences: list[str] | None = None,
        track_usage: bool = True,
    ) -> LLMResponse:
        """Send a message to the LLM and get a response.

        Args:
            model: Model name or alias
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            json_mode: Request JSON-formatted response
            stop_sequences: Sequences that will stop generation
            track_usage: Whether to track token usage

        Returns:
            LLMResponse with the model's response

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model not available
            APIError: For other API errors
        """
        client = await self._ensure_client()
        model_config = self._get_model_config(model)

        # Rate limiting
        await self._rate_limiter.acquire()

        # Build request
        body = self._build_request_body(
            model_config=model_config,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            stop_sequences=stop_sequences,
        )

        start_time = time.time()

        # Retry logic with tenacity
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_retries),
                wait=wait_exponential(multiplier=2, min=4, max=120),
                retry=retry_if_exception_type((RateLimitError, APIError)),
                reraise=True,
            ):
                with attempt:
                    attempt_num = attempt.retry_state.attempt_number
                    call_time = time.strftime("%H:%M:%S")
                    logger.info(
                        f"[{call_time}] API call to {model_config.name} "
                        f"(attempt {attempt_num}/{self._max_retries})"
                    )
                    response = await client.post(OPENROUTER_CHAT_ENDPOINT, json=body)
                    result = await self._handle_response(
                        response=response,
                        model_config=model_config,
                        start_time=start_time,
                        track_usage=track_usage,
                    )
                    end_time = time.strftime("%H:%M:%S")
                    logger.info(
                        f"[{end_time}] API call completed: {model_config.name} "
                        f"({result.latency_ms}ms)"
                    )
                    return result
        except RetryError:
            await self._usage_tracker.track_error()
            raise

        # This should never be reached, but satisfies type checker
        raise APIError("Unexpected error in send_message")

    async def _handle_response(
        self,
        response: httpx.Response,
        model_config: ModelConfig,
        start_time: float,
        track_usage: bool,
    ) -> LLMResponse:
        """Handle the API response.

        Args:
            response: HTTP response
            model_config: Model configuration
            start_time: Request start time
            track_usage: Whether to track usage

        Returns:
            LLMResponse

        Raises:
            Various LLMClientError subclasses
        """
        latency_ms = int((time.time() - start_time) * 1000)

        if response.status_code == 401:
            raise AuthenticationError("Invalid API key")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            logger.warning(
                f"[{time.strftime('%H:%M:%S')}] Rate limited! "
                f"Retry-After: {retry_after}s"
            )
            raise RateLimitError("Rate limit exceeded")
        elif response.status_code == 404:
            raise ModelNotFoundError("Model not found")
        elif response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", response.text)
            except Exception:
                error_msg = response.text
            raise APIError(error_msg, status_code=response.status_code)

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {e}")

        # Extract content
        choices = data.get("choices", [])
        if not choices:
            raise APIError("No choices in response")

        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        finish_reason = choice.get("finish_reason", "stop")

        # Extract usage
        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)

        cost = self._calculate_cost(model_config, prompt_tokens, completion_tokens)

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=model_config.name,
            cost_usd=cost,
        )

        # Track usage
        if track_usage:
            await self._usage_tracker.track(usage)

        return LLMResponse(
            content=content,
            model=data.get("model", model_config.name),
            usage=usage,
            finish_reason=finish_reason,
            raw_response=data,
            latency_ms=latency_ms,
        )

    async def send_message_stream(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Send a message and stream the response.

        Args:
            model: Model name or alias
            messages: List of conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Stop sequences

        Yields:
            Chunks of the response text

        Raises:
            Same exceptions as send_message
        """
        client = await self._ensure_client()
        model_config = self._get_model_config(model)

        await self._rate_limiter.acquire()

        body = self._build_request_body(
            model_config=model_config,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
        )
        body["stream"] = True

        async with client.stream("POST", OPENROUTER_CHAT_ENDPOINT, json=body) as response:
            if response.status_code >= 400:
                # Read full response for error message
                content = await response.aread()
                try:
                    error_data = json.loads(content)
                    error_msg = error_data.get("error", {}).get("message", content.decode())
                except Exception:
                    error_msg = content.decode()

                if response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code == 404:
                    raise ModelNotFoundError("Model not found")
                else:
                    raise APIError(error_msg, status_code=response.status_code)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def generate_documentation(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        json_mode: bool = True,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Convenience method for documentation generation.

        Wraps send_message with common settings for documentation tasks.

        Args:
            system_prompt: System prompt with instructions
            user_prompt: User prompt with code/context
            model: Model name or alias
            json_mode: Whether to request JSON output
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse
        """
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        return await self.send_message(
            model=model,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            temperature=0.0,  # Low temperature for documentation
        )

    async def validate_documentation(
        self,
        system_prompt: str,
        documentation: str,
        source_code: str,
        ground_truth: str | None,
        model: str,
    ) -> LLMResponse:
        """Convenience method for documentation validation.

        Args:
            system_prompt: System prompt with validation instructions
            documentation: Documentation to validate
            source_code: Original source code
            ground_truth: Static analysis ground truth (optional)
            model: Model name or alias

        Returns:
            LLMResponse with validation results
        """
        user_prompt = f"""## Documentation to Validate
{documentation}

## Source Code
```
{source_code}
```
"""
        if ground_truth:
            user_prompt += f"""
## Ground Truth (Static Analysis)
{ground_truth}
"""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        return await self.send_message(
            model=model,
            messages=messages,
            json_mode=True,
            temperature=0.0,
        )

    async def compare_documentation(
        self,
        system_prompt: str,
        stream_a_output: str,
        stream_b_output: str,
        ground_truth: str,
        model: str,
    ) -> LLMResponse:
        """Convenience method for documentation comparison.

        Args:
            system_prompt: System prompt with comparison instructions
            stream_a_output: Output from Stream A
            stream_b_output: Output from Stream B
            ground_truth: Static analysis ground truth
            model: Model name or alias (typically comparator model)

        Returns:
            LLMResponse with comparison results
        """
        user_prompt = f"""## Stream A Output
{stream_a_output}

## Stream B Output
{stream_b_output}

## Ground Truth (Static Analysis)
{ground_truth}
"""

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        return await self.send_message(
            model=model,
            messages=messages,
            json_mode=True,
            temperature=0.0,
        )

    @property
    def usage_tracker(self) -> UsageTracker:
        """Get the usage tracker instance."""
        return self._usage_tracker

    async def get_usage_summary(self) -> dict[str, Any]:
        """Get current usage summary.

        Returns:
            Dict with usage statistics
        """
        return await self._usage_tracker.get_summary()


# Convenience functions for getting configured clients


def get_documenter_client(
    stream_id: Literal["A", "B"],
    api_key: str | None = None,
) -> tuple[AsyncLLMClient, str]:
    """Get an LLM client configured for documenter role.

    Args:
        stream_id: Which stream (A or B)
        api_key: Optional API key override

    Returns:
        Tuple of (client, model_name)
    """
    from twinscribe.config.models import ModelsConfig

    models = ModelsConfig()
    if stream_id == "A":
        model_name = models.stream_a.documenter
    else:
        model_name = models.stream_b.documenter

    client = AsyncLLMClient(
        api_key=api_key,
        app_title=f"TwinScribe-Documenter-{stream_id}",
    )

    return client, model_name


def get_validator_client(
    stream_id: Literal["A", "B"],
    api_key: str | None = None,
) -> tuple[AsyncLLMClient, str]:
    """Get an LLM client configured for validator role.

    Args:
        stream_id: Which stream (A or B)
        api_key: Optional API key override

    Returns:
        Tuple of (client, model_name)
    """
    from twinscribe.config.models import ModelsConfig

    models = ModelsConfig()
    if stream_id == "A":
        model_name = models.stream_a.validator
    else:
        model_name = models.stream_b.validator

    client = AsyncLLMClient(
        api_key=api_key,
        app_title=f"TwinScribe-Validator-{stream_id}",
    )

    return client, model_name


def get_comparator_client(
    api_key: str | None = None,
) -> tuple[AsyncLLMClient, str]:
    """Get an LLM client configured for comparator role.

    Args:
        api_key: Optional API key override

    Returns:
        Tuple of (client, model_name)
    """
    from twinscribe.config.models import ModelsConfig

    models = ModelsConfig()
    model_name = models.comparator

    client = AsyncLLMClient(
        api_key=api_key,
        app_title="TwinScribe-Comparator",
    )

    return client, model_name
