"""
Configuration Loader.

Loads and validates configuration from YAML files with environment
variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError

from twinscribe.config.environment import load_environment
from twinscribe.config.models import TwinscribeConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        errors: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.errors = errors or []


class ConfigLoader:
    """Loads configuration from YAML files.

    Supports:
    - YAML configuration files
    - Environment variable substitution
    - Defaults from environment
    - Validation via Pydantic

    Usage:
        loader = ConfigLoader("config.yaml")
        config = loader.load()
    """

    # Pattern for environment variable substitution
    ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        env_file: str = ".env",
    ) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to YAML config file (optional)
            env_file: Path to .env file for environment loading
        """
        self._config_path = Path(config_path) if config_path else None
        self._env_file = env_file
        self._raw_config: Optional[dict] = None
        self._config: Optional[TwinscribeConfig] = None

    @property
    def config_path(self) -> Optional[Path]:
        """Get config file path."""
        return self._config_path

    @property
    def config(self) -> Optional[TwinscribeConfig]:
        """Get loaded configuration."""
        return self._config

    def load(self) -> TwinscribeConfig:
        """Load and validate configuration.

        Returns:
            Validated TwinscribeConfig

        Raises:
            ConfigurationError: If config is invalid
            FileNotFoundError: If config file not found
        """
        # Load environment first
        load_environment(self._env_file)

        # Load YAML if path provided
        if self._config_path:
            self._raw_config = self._load_yaml()
        else:
            self._raw_config = {}

        # Substitute environment variables
        processed = self._substitute_env_vars(self._raw_config)

        # Validate and create config
        try:
            self._config = TwinscribeConfig(**processed)
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation failed: {e.error_count()} errors",
                errors=e.errors(),
            )

        return self._config

    def _load_yaml(self) -> dict[str, Any]:
        """Load YAML configuration file.

        Returns:
            Parsed YAML as dict

        Raises:
            FileNotFoundError: If file doesn't exist
            ConfigurationError: If YAML is invalid
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        try:
            with open(self._config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}")

    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in config.

        Supports syntax:
        - ${VAR_NAME} - substitute with env var, error if not set
        - ${VAR_NAME:default} - substitute with default if not set

        Args:
            data: Configuration data (dict, list, or scalar)

        Returns:
            Data with environment variables substituted
        """
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_string(data)
        else:
            return data

    def _substitute_string(self, value: str) -> str:
        """Substitute environment variables in a string.

        Args:
            value: String potentially containing ${VAR} patterns

        Returns:
            String with variables substituted
        """
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                # Return original if not found (will fail validation if required)
                return match.group(0)

        return self.ENV_PATTERN.sub(replace, value)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to YAML file.

        Args:
            path: Path to save to (defaults to original config_path)

        Raises:
            ValueError: If no config loaded or no path specified
        """
        if self._config is None:
            raise ValueError("No configuration loaded")

        save_path = Path(path) if path else self._config_path
        if save_path is None:
            raise ValueError("No path specified for saving")

        with open(save_path, "w") as f:
            yaml.safe_dump(
                self._config.to_yaml_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )


# Global configuration instance
_global_config: Optional[TwinscribeConfig] = None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_file: str = ".env",
) -> TwinscribeConfig:
    """Load configuration from file.

    Convenience function for loading configuration.

    Args:
        config_path: Path to YAML config file
        env_file: Path to .env file

    Returns:
        Validated TwinscribeConfig
    """
    global _global_config

    loader = ConfigLoader(config_path, env_file)
    _global_config = loader.load()
    return _global_config


def get_config() -> TwinscribeConfig:
    """Get the global configuration.

    Returns:
        Global TwinscribeConfig

    Raises:
        RuntimeError: If configuration not loaded
    """
    if _global_config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _global_config


def reset_config() -> None:
    """Reset global configuration.

    Useful for testing.
    """
    global _global_config
    _global_config = None


def create_default_config(codebase_path: str) -> TwinscribeConfig:
    """Create a configuration with defaults.

    Useful for programmatic configuration without a YAML file.

    Args:
        codebase_path: Path to the codebase

    Returns:
        TwinscribeConfig with defaults
    """
    from twinscribe.config.models import CodebaseConfig

    return TwinscribeConfig(
        codebase=CodebaseConfig(path=codebase_path)
    )
