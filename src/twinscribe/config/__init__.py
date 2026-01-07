"""
Dual-Stream Documentation System - Configuration Management

This module provides configuration management including:
- YAML configuration loading and validation
- Environment variable handling
- Model configuration with OpenRouter support
- Configuration defaults and overrides
- CrossCheck verification framework configuration
"""

from twinscribe.config.environment import (
    EnvironmentConfig,
    ensure_dotenv_loaded,
    get_api_key,
    get_env_model,
    load_environment,
    reset_environment,
)
from twinscribe.config.loader import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATHS,
    ENV_VAR_OVERRIDES,
    ConfigLoader,
    ConfigurationError,
    create_default_config,
    get_config,
    get_loader,
    load_config,
    load_config_from_env,
    reload_config,
    reset_config,
)
from twinscribe.config.models import (
    AdversarialReviewConfig,
    BeadsConfig,
    CodebaseConfig,
    ConvergenceConfig,
    LoggingConfig,
    LogLevel,
    MaskedReconstructionConfig,
    MaskType,
    ModelConfig,
    ModelsConfig,
    MutationDetectionConfig,
    MutationType,
    OutputConfig,
    QAInterrogationConfig,
    QAQuestionCategory,
    ScenarioType,
    ScenarioWalkthroughConfig,
    StaticAnalysisConfig,
    StreamModelsConfig,
    TestGenerationConfig,
    TwinscribeConfig,
    VerificationConfig,
    # Verification configuration
    VerificationStrategy,
    VerificationThresholdsConfig,
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
    "LogLevel",
    "LoggingConfig",
    "TwinscribeConfig",
    # Verification config models
    "VerificationStrategy",
    "VerificationThresholdsConfig",
    "QAQuestionCategory",
    "QAInterrogationConfig",
    "MaskType",
    "MaskedReconstructionConfig",
    "ScenarioType",
    "ScenarioWalkthroughConfig",
    "MutationType",
    "MutationDetectionConfig",
    "AdversarialReviewConfig",
    "TestGenerationConfig",
    "VerificationConfig",
    # Loader
    "ConfigLoader",
    "ConfigurationError",
    "load_config",
    "load_config_from_env",
    "get_config",
    "reload_config",
    "reset_config",
    "create_default_config",
    "get_loader",
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_PATHS",
    "ENV_VAR_OVERRIDES",
    # Environment
    "EnvironmentConfig",
    "load_environment",
    "ensure_dotenv_loaded",
    "get_env_model",
    "get_api_key",
    "reset_environment",
]
