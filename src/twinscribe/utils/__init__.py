"""
TwinScribe Utilities Module.

This module provides shared utility functions and helpers:

- Configuration loading and validation
- Logging setup with Rich formatting
- Async utilities and retry decorators
- File and path handling helpers
- Token counting and cost estimation
- LLM client for OpenRouter API

These utilities are used across all other modules.
"""

from twinscribe.utils.llm_client import (
    AsyncLLMClient,
    LLMClientError,
    AuthenticationError,
    RateLimitError,
    ModelNotFoundError,
    APIError,
    TokenUsage,
    LLMResponse,
    Message,
    UsageTracker,
    RateLimiter,
    get_documenter_client,
    get_validator_client,
    get_comparator_client,
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
]
