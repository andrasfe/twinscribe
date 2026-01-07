"""
TwinScribe Utilities Module.

This module provides shared utility functions and helpers:

- Configuration loading and validation
- Logging setup with Rich formatting
- Async utilities and retry decorators
- File and path handling helpers
- Token counting and cost estimation
- LLM client for OpenRouter API
- Metrics collection and structured logging

These utilities are used across all other modules.
"""

from twinscribe.utils.llm_client import (
    APIError,
    AsyncLLMClient,
    AuthenticationError,
    LLMClientError,
    LLMResponse,
    Message,
    ModelNotFoundError,
    RateLimiter,
    RateLimitError,
    TokenUsage,
    UsageTracker,
    get_comparator_client,
    get_documenter_client,
    get_validator_client,
)
from twinscribe.utils.metrics import (
    BeadsTicketMetrics,
    CallGraphMetrics,
    ComponentMetrics,
    ConvergenceMetrics,
    CostMetrics,
    MetricCategory,
    MetricsCollector,
    # Metrics classes
    PhaseMetrics,
    # Enums
    ProcessingPhase,
    # Structured logging
    StructuredLogger,
    # Factory functions
    get_metrics_collector,
    get_structured_logger,
)

__all__ = [
    # LLM Client
    "AsyncLLMClient",
    "LLMClientError",
    "AuthenticationError",
    "RateLimitError",
    "ModelNotFoundError",
    "APIError",
    "TokenUsage",
    "LLMResponse",
    "Message",
    "UsageTracker",
    "RateLimiter",
    # Convenience functions
    "get_documenter_client",
    "get_validator_client",
    "get_comparator_client",
    # Metrics - Enums
    "ProcessingPhase",
    "MetricCategory",
    # Metrics - Classes
    "PhaseMetrics",
    "CallGraphMetrics",
    "CostMetrics",
    "ConvergenceMetrics",
    "BeadsTicketMetrics",
    "ComponentMetrics",
    "MetricsCollector",
    # Structured Logging
    "StructuredLogger",
    # Metrics Factory Functions
    "get_metrics_collector",
    "get_structured_logger",
]
