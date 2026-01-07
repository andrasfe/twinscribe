"""
Base agent classes and configurations.

Provides the foundational abstract classes and common functionality
for all agents in the documentation system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from twinscribe.models.base import ModelTier, StreamId


class AgentConfig(BaseModel):
    """Configuration for an agent.

    Attributes:
        agent_id: Unique identifier for this agent instance (A1, A2, B1, B2, C)
        stream_id: Which stream this agent belongs to (A, B, or None for comparator)
        model_tier: Cost tier of the model
        provider: Model provider (anthropic, openai)
        model_name: Full model identifier
        cost_per_million_input: Cost per million input tokens
        cost_per_million_output: Cost per million output tokens
        max_tokens: Maximum tokens per request
        temperature: Sampling temperature
        timeout_seconds: Request timeout
        max_retries: Maximum retry attempts
    """

    agent_id: str = Field(..., description="Agent identifier")
    stream_id: StreamId | None = Field(
        default=None,
        description="Stream identifier (None for comparator)",
    )
    model_tier: ModelTier = Field(..., description="Model cost tier")
    provider: str = Field(
        ...,
        description="Model provider",
        examples=["anthropic", "openai"],
    )
    model_name: str = Field(
        ...,
        description="Model identifier",
        examples=["claude-sonnet-4-5-20250929", "gpt-4o"],
    )
    cost_per_million_input: float = Field(
        default=0.0,
        ge=0.0,
        description="Input token cost per million",
    )
    cost_per_million_output: float = Field(
        default=0.0,
        ge=0.0,
        description="Output token cost per million",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Max tokens per response",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Request timeout",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retry attempts",
    )


@dataclass
class AgentMetrics:
    """Metrics collected during agent execution.

    Attributes:
        requests_made: Number of API requests
        tokens_input: Total input tokens
        tokens_output: Total output tokens
        cost_total: Total cost in USD
        errors: Number of errors encountered
        avg_latency_ms: Average request latency
        started_at: When processing started
        completed_at: When processing completed
    """

    requests_made: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cost_total: float = 0.0
    errors: int = 0
    avg_latency_ms: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    _latencies: list[float] = field(default_factory=list)

    def record_request(
        self,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float,
    ) -> None:
        """Record metrics from a single request.

        Args:
            input_tokens: Tokens in request
            output_tokens: Tokens in response
            latency_ms: Request latency
            cost: Request cost in USD
        """
        self.requests_made += 1
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.cost_total += cost
        self._latencies.append(latency_ms)
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.errors += 1

    @property
    def duration_seconds(self) -> float | None:
        """Total processing duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Generic type variables for input/output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all agents.

    Provides common functionality including:
    - Configuration management
    - Metrics collection
    - Error handling with retries
    - Async execution interface

    Type Parameters:
        InputT: Type of input to the agent
        OutputT: Type of output from the agent
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent.

        Args:
            config: Agent configuration
        """
        self._config = config
        self._metrics = AgentMetrics()
        self._initialized = False

    @property
    def config(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config

    @property
    def metrics(self) -> AgentMetrics:
        """Get agent metrics."""
        return self._metrics

    @property
    def agent_id(self) -> str:
        """Get agent identifier."""
        return self._config.agent_id

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent (e.g., connect to API, load resources).

        Called once before first use. Must be called before process().

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """Process input and produce output.

        Main processing method that must be implemented by subclasses.

        Args:
            input_data: Input to process

        Returns:
            Processed output

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
            Exception: On processing error
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources and shutdown the agent.

        Called when agent is no longer needed.
        """
        pass

    async def process_batch(
        self,
        inputs: list[InputT],
        concurrency: int = 5,
    ) -> list[OutputT]:
        """Process multiple inputs with controlled concurrency.

        Default implementation processes sequentially. Subclasses may
        override for parallel processing.

        Args:
            inputs: List of inputs to process
            concurrency: Maximum concurrent requests (for subclass use)

        Returns:
            List of outputs in same order as inputs
        """
        results = []
        for input_data in inputs:
            result = await self.process(input_data)
            results.append(result)
        return results

    def reset_metrics(self) -> None:
        """Reset collected metrics."""
        self._metrics = AgentMetrics()

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = input_tokens * self._config.cost_per_million_input / 1_000_000
        output_cost = output_tokens * self._config.cost_per_million_output / 1_000_000
        return input_cost + output_cost


class LLMClient(ABC):
    """Abstract interface for LLM API clients.

    Provides a common interface for different LLM providers
    (Anthropic, OpenAI via OpenRouter).
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> "LLMResponse":
        """Generate a completion.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            response_format: Optional JSON schema for structured output

        Returns:
            LLM response with content and usage info
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass


@dataclass
class LLMResponse:
    """Response from an LLM API call.

    Attributes:
        content: Response text content
        input_tokens: Tokens in request
        output_tokens: Tokens in response
        model: Model used
        latency_ms: Request latency
        finish_reason: Why generation stopped
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    latency_ms: float
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens
