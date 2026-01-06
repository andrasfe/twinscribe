"""
Dual-Stream Documentation System - Configuration Management

This module provides configuration management including:
- YAML configuration loading and validation
- Environment variable handling
- Model configuration with OpenRouter support
- Configuration defaults and overrides
"""

from twinscribe.config.models import (
    CodebaseConfig,
    ModelConfig,
    StreamModelsConfig,
    ModelsConfig,
    ConvergenceConfig,
    BeadsConfig,
    StaticAnalysisConfig,
    OutputConfig,
    TwinscribeConfig,
)
from twinscribe.config.loader import (
    ConfigLoader,
    load_config,
    get_config,
)
from twinscribe.config.environment import (
    EnvironmentConfig,
    load_environment,
    get_api_key,
)

__all__ = [
    # Config models
    "CodebaseConfig",
    "ModelConfig",
    "StreamModelsConfig",
    "ModelsConfig",
    "ConvergenceConfig",
    "BeadsConfig",
    "StaticAnalysisConfig",
    "OutputConfig",
    "TwinscribeConfig",
    # Loader
    "ConfigLoader",
    "load_config",
    "get_config",
    # Environment
    "EnvironmentConfig",
    "load_environment",
    "get_api_key",
]
