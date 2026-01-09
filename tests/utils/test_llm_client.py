"""
Tests for the OpenRouter LLM Client.

These tests use mocking to avoid actual API calls.
"""

import asyncio
import json

import pytest
import respx
from httpx import Response

from twinscribe.utils.llm_client import (
    OPENROUTER_BASE_URL,
    OPENROUTER_CHAT_ENDPOINT,
    AsyncLLMClient,
    AuthenticationError,
    LLMResponse,
    Message,
    RateLimiter,
    RateLimitError,
    TokenUsage,
    UsageTracker,
)


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role="system", content="You are helpful")
        result = msg.to_dict()
        assert result == {"role": "system", "content": "You are helpful"}


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_token_usage_defaults(self):
        """Test TokenUsage with default values."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cost_usd == 0.0

    def test_token_usage_add(self):
        """Test adding two TokenUsage objects."""
        usage1 = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            cost_usd=0.01,
        )
        usage2 = TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            model="gpt-4o",
            cost_usd=0.02,
        )
        result = usage1.add(usage2)

        assert result.prompt_tokens == 300
        assert result.completion_tokens == 150
        assert result.total_tokens == 450
        assert result.cost_usd == 0.03


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_llm_response_creation(self):
        """Test creating an LLM response."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4o",
            usage=usage,
            finish_reason="stop",
            latency_ms=500,
        )

        assert response.content == "Hello, world!"
        assert response.model == "gpt-4o"
        assert response.latency_ms == 500

    def test_llm_response_to_json(self):
        """Test converting response to JSON."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        response = LLMResponse(
            content="Test",
            model="gpt-4o",
            usage=usage,
        )
        result = response.to_json()

        assert result["content"] == "Test"
        assert result["model"] == "gpt-4o"
        assert result["usage"]["prompt_tokens"] == 100


class TestUsageTracker:
    """Tests for UsageTracker class."""

    @pytest.mark.asyncio
    async def test_track_single_usage(self):
        """Test tracking a single usage."""
        tracker = UsageTracker()
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            cost_usd=0.01,
        )
        await tracker.track(usage)

        summary = await tracker.get_summary()
        assert summary["totals"]["requests"] == 1
        assert summary["totals"]["prompt_tokens"] == 100
        assert summary["totals"]["completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_track_multiple_models(self):
        """Test tracking usage across multiple models."""
        tracker = UsageTracker()

        usage1 = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-4o",
            cost_usd=0.01,
        )
        usage2 = TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            model="claude-sonnet-4-5",
            cost_usd=0.02,
        )

        await tracker.track(usage1)
        await tracker.track(usage2)

        summary = await tracker.get_summary()
        assert summary["totals"]["requests"] == 2
        assert "gpt-4o" in summary["by_model"]
        assert "claude-sonnet-4-5" in summary["by_model"]

    @pytest.mark.asyncio
    async def test_track_error(self):
        """Test tracking errors."""
        tracker = UsageTracker()
        await tracker.track_error()
        await tracker.track_error()

        summary = await tracker.get_summary()
        assert summary["totals"]["errors"] == 2

    def test_reset(self):
        """Test resetting the tracker."""
        tracker = UsageTracker()
        tracker._total_requests = 10
        tracker.reset()

        assert tracker._total_requests == 0
        assert len(tracker._usage_by_model) == 0


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        """Test acquiring when under limit."""
        limiter = RateLimiter(requests_per_minute=60)

        # Should not block
        start = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        """Test acquiring multiple tokens."""
        limiter = RateLimiter(requests_per_minute=60)

        # Acquire several quickly
        for _ in range(5):
            await limiter.acquire()

        # Should still work (tokens refill over time)
        await limiter.acquire()


@pytest.mark.unit
class TestAsyncLLMClient:
    """Tests for AsyncLLMClient class."""

    def test_client_creation(self):
        """Test creating a client."""
        client = AsyncLLMClient(
            api_key="test-key",
            timeout=30.0,
            max_retries=5,
        )

        assert client._api_key == "test-key"
        assert client._timeout == 30.0
        assert client._max_retries == 5

    def test_get_model_config(self):
        """Test getting model configuration."""
        client = AsyncLLMClient(api_key="test-key")

        config = client._get_model_config("gpt-4o")
        assert config.name == "gpt-4o"

        config = client._get_model_config("claude-sonnet-4-5")
        assert config.name == "claude-3-5-sonnet-20241022"

    def test_get_model_config_unknown(self):
        """Test getting unknown model configuration.

        Unknown models now return a dynamically created config
        instead of raising ModelNotFoundError.
        """
        client = AsyncLLMClient(api_key="test-key")

        # Unknown models get a dynamic config for OpenRouter
        config = client._get_model_config("unknown-model")
        assert config.name == "unknown-model"
        assert config.provider.value == "openrouter"

    def test_calculate_cost(self):
        """Test cost calculation."""
        client = AsyncLLMClient(api_key="test-key")
        config = client._get_model_config("gpt-4o")

        # gpt-4o: $2.50/M input, $10/M output
        cost = client._calculate_cost(
            model_config=config,
            prompt_tokens=1_000_000,
            completion_tokens=500_000,
        )

        expected = 2.50 + 5.0  # $2.50 for input + $5 for output
        assert abs(cost - expected) < 0.01

    def test_build_request_body(self):
        """Test building request body."""
        client = AsyncLLMClient(api_key="test-key")
        config = client._get_model_config("gpt-4o")

        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
        ]

        body = client._build_request_body(
            model_config=config,
            messages=messages,
            max_tokens=1000,
            temperature=0.5,
        )

        assert body["model"] == "openai/gpt-4o"
        assert len(body["messages"]) == 2
        assert body["max_tokens"] == 1000
        assert body["temperature"] == 0.5


@pytest.mark.llm
class TestAsyncLLMClientAPI:
    """Tests for AsyncLLMClient API interactions (mocked)."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_success(self):
        """Test successful API call."""
        # Mock the OpenRouter API response
        mock_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": {"content": "Hello! How can I help you?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
            },
        }

        respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncLLMClient(api_key="test-key") as client:
            response = await client.send_message(
                model="gpt-4o",
                messages=[Message(role="user", content="Hello")],
            )

        assert response.content == "Hello! How can I help you?"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_auth_error(self):
        """Test authentication error handling."""
        respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(401, json={"error": {"message": "Invalid API key"}})
        )

        async with AsyncLLMClient(api_key="invalid-key") as client:
            with pytest.raises(AuthenticationError):
                await client.send_message(
                    model="gpt-4o",
                    messages=[Message(role="user", content="Hello")],
                )

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_rate_limit(self):
        """Test rate limit error handling."""
        respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(429, json={"error": {"message": "Rate limit"}})
        )

        async with AsyncLLMClient(api_key="test-key", max_retries=1) as client:
            with pytest.raises(RateLimitError):
                await client.send_message(
                    model="gpt-4o",
                    messages=[Message(role="user", content="Hello")],
                )

    @pytest.mark.asyncio
    @respx.mock
    async def test_send_message_json_mode(self):
        """Test JSON mode request."""
        mock_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": {"content": '{"result": "success"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        route = respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncLLMClient(api_key="test-key") as client:
            response = await client.send_message(
                model="gpt-4o",
                messages=[Message(role="user", content="Return JSON")],
                json_mode=True,
            )

        # Verify json_mode was sent in request
        request_body = json.loads(route.calls.last.request.content)
        assert request_body.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_generate_documentation(self):
        """Test documentation generation convenience method."""
        mock_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": {"content": '{"summary": "Test function", "parameters": []}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }

        respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncLLMClient(api_key="test-key") as client:
            response = await client.generate_documentation(
                system_prompt="Document this code",
                user_prompt="def hello(): pass",
                model="gpt-4o",
            )

        assert "summary" in response.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_usage_tracking(self):
        """Test that usage is properly tracked."""
        mock_response = {
            "id": "gen-123",
            "model": "openai/gpt-4o",
            "choices": [
                {
                    "message": {"content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
            return_value=Response(200, json=mock_response)
        )

        async with AsyncLLMClient(api_key="test-key") as client:
            await client.send_message(
                model="gpt-4o",
                messages=[Message(role="user", content="Hello")],
            )
            await client.send_message(
                model="gpt-4o",
                messages=[Message(role="user", content="World")],
            )

            summary = await client.get_usage_summary()

        assert summary["totals"]["requests"] == 2
        assert summary["totals"]["prompt_tokens"] == 200
        assert summary["totals"]["completion_tokens"] == 100


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client as async context manager."""
    async with AsyncLLMClient(api_key="test-key") as client:
        assert client._client is not None

    assert client._client is None  # Closed after exit
