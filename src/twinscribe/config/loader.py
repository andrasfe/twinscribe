"""
Configuration Loader.

Loads and validates configuration from YAML files with environment
variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from twinscribe.config.environment import load_environment
from twinscribe.config.models import TwinscribeConfig

# Default configuration file locations
DEFAULT_CONFIG_PATHS = [
    "config.yaml",
    "config.yml",
    "twinscribe.yaml",
    "twinscribe.yml",
    ".twinscribe.yaml",
    ".twinscribe.yml",
]

# Environment variable for config path
CONFIG_ENV_VAR = "TWINSCRIBE_CONFIG"

# Environment variable overrides for configuration settings
# Maps env var name to config path (dot-separated)
ENV_VAR_OVERRIDES = {
    # Model settings
    "TWINSCRIBE_MODEL_PROVIDER": "models.stream_a.documenter",
    "TWINSCRIBE_STREAM_A_DOCUMENTER": "models.stream_a.documenter",
    "TWINSCRIBE_STREAM_A_VALIDATOR": "models.stream_a.validator",
    "TWINSCRIBE_STREAM_B_DOCUMENTER": "models.stream_b.documenter",
    "TWINSCRIBE_STREAM_B_VALIDATOR": "models.stream_b.validator",
    "TWINSCRIBE_COMPARATOR": "models.comparator",
    # Convergence settings
    "TWINSCRIBE_MAX_ITERATIONS": "convergence.max_iterations",
    "TWINSCRIBE_CONVERGENCE_THRESHOLD": "convergence.call_graph_match_threshold",
    # Output settings
    "TWINSCRIBE_OUTPUT_DIR": "output.base_dir",
    # Logging settings
    "TWINSCRIBE_LOG_LEVEL": "logging.level",
    "TWINSCRIBE_LOG_FILE": "logging.file",
    # Runtime flags
    "TWINSCRIBE_DEBUG": "debug",
    "TWINSCRIBE_DRY_RUN": "dry_run",
}


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""

    def __init__(
        self,
        message: str,
        errors: list[dict] | None = None,
        path: Path | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            message: Error message
            errors: List of validation errors (from Pydantic)
            path: Path to the config file that caused the error
        """
        super().__init__(message)
        self.errors = errors or []
        self.path = path

    def __str__(self) -> str:
        """Format error message with details."""
        msg = super().__str__()
        if self.path:
            msg = f"{msg} (file: {self.path})"
        if self.errors:
            error_details = []
            for err in self.errors[:5]:  # Show first 5 errors
                loc = ".".join(str(x) for x in err.get("loc", []))
                error_msg = err.get("msg", "Unknown error")
                error_details.append(f"  - {loc}: {error_msg}")
            if len(self.errors) > 5:
                error_details.append(f"  ... and {len(self.errors) - 5} more errors")
            msg = f"{msg}\n" + "\n".join(error_details)
        return msg


class ConfigLoader:
    """Loads configuration from YAML files.

    Supports:
    - YAML configuration files
    - Environment variable substitution (${VAR} and ${VAR:-default} syntax)
    - Defaults from environment
    - Validation via Pydantic
    - Configuration caching and reloading

    Usage:
        # Load from specific file
        loader = ConfigLoader("config.yaml")
        config = loader.load()

        # Load from environment variable or default locations
        loader = ConfigLoader()
        config = loader.load_from_env()

        # Get cached configuration
        config = loader.get()

        # Reload configuration
        config = loader.reload()
    """

    # Pattern for environment variable substitution
    # Matches: ${VAR_NAME} or ${VAR_NAME:-default_value} or ${VAR_NAME:default_value}
    # Both :- and : syntax are supported for default values
    # The dash in :- is optional (shell-style vs simple syntax)
    ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-?([^}]*))?\}")

    def __init__(
        self,
        config_path: str | Path | None = None,
        env_file: str = ".env",
    ) -> None:
        """Initialize the config loader.

        Args:
            config_path: Path to YAML config file (optional).
                If not provided, use load_from_env() to auto-discover.
            env_file: Path to .env file for environment loading
        """
        self._config_path = Path(config_path) if config_path else None
        self._env_file = env_file
        self._raw_config: dict[str, Any] | None = None
        self._config: TwinscribeConfig | None = None
        self._loaded_from_path: Path | None = None

    @property
    def config_path(self) -> Path | None:
        """Get config file path."""
        return self._config_path

    @property
    def loaded_from_path(self) -> Path | None:
        """Get the path the config was actually loaded from.

        This may differ from config_path when using load_from_env().
        """
        return self._loaded_from_path

    @property
    def config(self) -> TwinscribeConfig | None:
        """Get loaded configuration.

        Returns:
            The loaded TwinscribeConfig or None if not loaded yet.
        """
        return self._config

    def get(self) -> TwinscribeConfig:
        """Get the loaded configuration.

        Returns:
            The loaded TwinscribeConfig.

        Raises:
            RuntimeError: If configuration has not been loaded yet.
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded. Call load() or load_from_env() first.")
        return self._config

    def load(self, path: str | Path | None = None) -> TwinscribeConfig:
        """Load and validate configuration from a specific path.

        Args:
            path: Optional path to config file. If provided, overrides
                the path set in __init__. If not provided and no path
                was set in __init__, loads with empty config (uses defaults).

        Returns:
            Validated TwinscribeConfig

        Raises:
            ConfigurationError: If config is invalid
            FileNotFoundError: If config file not found
        """
        # Override path if provided
        if path is not None:
            self._config_path = Path(path)

        # Load environment first (loads .env file if present)
        load_environment(self._env_file)

        # Load YAML if path provided
        if self._config_path:
            self._raw_config = self._load_yaml()
            self._loaded_from_path = self._config_path
        else:
            self._raw_config = {}
            self._loaded_from_path = None

        # Substitute environment variables in YAML values (${VAR} syntax)
        processed = self._substitute_env_vars(self._raw_config)

        # Apply explicit environment variable overrides (TWINSCRIBE_* vars)
        processed = self._apply_env_overrides(processed)

        # Clean up None values that should be omitted
        # (YAML parses empty sections as None, but Pydantic expects them to be omitted)
        processed = self._clean_none_values(processed)

        # Validate and create config
        try:
            self._config = TwinscribeConfig(**processed)
        except ValidationError as e:
            raise ConfigurationError(
                f"Configuration validation failed: {e.error_count()} errors",
                errors=e.errors(),
                path=self._loaded_from_path,
            )

        return self._config

    def load_from_env(self) -> TwinscribeConfig:
        """Load configuration from environment variable or default locations.

        This method searches for configuration in the following order:
        1. TWINSCRIBE_CONFIG environment variable (if set)
        2. Default config file locations in current directory:
           - config.yaml, config.yml
           - twinscribe.yaml, twinscribe.yml
           - .twinscribe.yaml, .twinscribe.yml

        Returns:
            Validated TwinscribeConfig

        Raises:
            ConfigurationError: If config is invalid
            FileNotFoundError: If no config file found and no defaults work
        """
        # Load environment first
        load_environment(self._env_file)

        # Check environment variable first
        env_config_path = os.environ.get(CONFIG_ENV_VAR)
        if env_config_path:
            config_path = Path(env_config_path)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file specified by {CONFIG_ENV_VAR} not found: {env_config_path}"
                )
            self._config_path = config_path
            return self.load()

        # Search default locations
        for default_path in DEFAULT_CONFIG_PATHS:
            path = Path(default_path)
            if path.exists():
                self._config_path = path
                return self.load()

        # No config file found - this is an error since codebase.path is required
        searched_paths = ", ".join(DEFAULT_CONFIG_PATHS)
        raise FileNotFoundError(
            f"No configuration file found. Searched: {searched_paths}. "
            f"Set {CONFIG_ENV_VAR} environment variable or create a config file."
        )

    def reload(self) -> TwinscribeConfig:
        """Reload configuration from the previously loaded path.

        This is useful when the configuration file has been modified
        and you want to pick up the changes without restarting.

        Returns:
            Reloaded and validated TwinscribeConfig

        Raises:
            RuntimeError: If no configuration was previously loaded
            ConfigurationError: If reloaded config is invalid
            FileNotFoundError: If config file no longer exists
        """
        if self._loaded_from_path is None and self._config_path is None:
            raise RuntimeError(
                "Cannot reload: no configuration path. Call load() or load_from_env() first."
            )

        # Clear cached config
        self._raw_config = None
        self._config = None

        # Reload from the path we loaded from (or config_path if not yet loaded)
        reload_path = self._loaded_from_path or self._config_path
        if reload_path:
            self._config_path = reload_path

        return self.load()

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
            with open(self._config_path) as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {e}")

    def _substitute_env_vars(self, data: Any, parent_key: str = "") -> Any:
        """Recursively substitute environment variables in config.

        Supports syntax:
        - ${VAR_NAME} - substitute with env var, error if not set
        - ${VAR_NAME:default} - substitute with default if not set

        Args:
            data: Configuration data (dict, list, or scalar)
            parent_key: The parent key path (for context in transformations)

        Returns:
            Data with environment variables substituted
        """
        if isinstance(data, dict):
            return {
                k: self._substitute_env_vars(v, f"{parent_key}.{k}" if parent_key else k)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._substitute_env_vars(item, parent_key) for item in data]
        elif isinstance(data, str):
            return self._substitute_string(data)
        elif data is None:
            # Return None as-is; will be cleaned up by _clean_none_values
            return None
        else:
            return data

    def _clean_none_values(self, data: Any) -> Any:
        """Recursively remove None values from nested dicts.

        YAML parses empty sections (with only comments) as None,
        but Pydantic expects them to be omitted so it can use defaults.

        Args:
            data: Configuration data (dict, list, or scalar)

        Returns:
            Data with None dict values removed (allowing Pydantic defaults)
        """
        if isinstance(data, dict):
            return {k: self._clean_none_values(v) for k, v in data.items() if v is not None}
        elif isinstance(data, list):
            return [self._clean_none_values(item) for item in data]
        else:
            return data

    def _substitute_string(self, value: str) -> Any:
        """Substitute environment variables in a string.

        Handles type coercion for special values:
        - "true"/"false" -> boolean
        - numeric strings -> int or float
        - empty string -> None (for optional fields)

        Args:
            value: String potentially containing ${VAR} patterns

        Returns:
            String with variables substituted, or coerced type
        """
        # Check if the entire string is a single env var reference
        full_match = self.ENV_PATTERN.fullmatch(value)
        if full_match:
            var_name = full_match.group(1)
            default = full_match.group(2)

            env_value = os.environ.get(var_name)
            resolved = env_value if env_value is not None else default

            if resolved is not None:
                return self._coerce_type(resolved)
            else:
                # Return original if not found (will fail validation if required)
                return value

        # For strings with embedded env vars, do simple string substitution
        def replace(match: re.Match[str]) -> str:
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

    def _coerce_type(self, value: str) -> Any:
        """Coerce string value to appropriate Python type.

        Args:
            value: String value to coerce

        Returns:
            Coerced value (bool, int, float, None, or original string)
        """
        # Handle empty string
        if value == "":
            return None

        # Handle boolean values (case-insensitive)
        lower_value = value.lower()
        if lower_value in ("true", "yes", "on", "1"):
            return True
        if lower_value in ("false", "no", "off", "0"):
            return False

        # Handle numeric values
        try:
            # Try integer first
            if "." not in value and "e" not in value.lower():
                return int(value)
            # Then try float
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _apply_env_overrides(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables in ENV_VAR_OVERRIDES take precedence over
        values in the config file.

        Args:
            config_dict: Configuration dictionary from YAML

        Returns:
            Configuration with overrides applied
        """
        for env_var, config_path in ENV_VAR_OVERRIDES.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Coerce the value to appropriate type
                coerced_value = self._coerce_type(env_value)

                # Navigate to the right place in the config dict
                self._set_nested_value(config_dict, config_path, coerced_value)

        return config_dict

    def _set_nested_value(
        self,
        config_dict: dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        """Set a nested value in a dictionary using dot notation.

        Args:
            config_dict: Configuration dictionary
            path: Dot-separated path (e.g., "models.stream_a.documenter")
            value: Value to set
        """
        parts = path.split(".")
        current = config_dict

        # Navigate/create nested dicts for all but the last part
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    def save(self, path: str | Path | None = None) -> None:
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


# Global loader instance for caching
_global_loader: ConfigLoader | None = None
_global_config: TwinscribeConfig | None = None


def load_config(
    config_path: str | Path | None = None,
    env_file: str = ".env",
) -> TwinscribeConfig:
    """Load configuration from file.

    Convenience function for loading configuration from a specific path.

    Args:
        config_path: Path to YAML config file. If None, an empty config
            will be attempted (will fail if required fields like codebase.path
            are not provided).
        env_file: Path to .env file

    Returns:
        Validated TwinscribeConfig

    Raises:
        ConfigurationError: If config validation fails
        FileNotFoundError: If config file not found
    """
    global _global_loader, _global_config

    _global_loader = ConfigLoader(config_path, env_file)
    _global_config = _global_loader.load()
    return _global_config


def load_config_from_env(env_file: str = ".env") -> TwinscribeConfig:
    """Load configuration from environment variable or default locations.

    Convenience function that searches for configuration in:
    1. TWINSCRIBE_CONFIG environment variable (if set)
    2. Default config file locations (config.yaml, twinscribe.yaml, etc.)

    Args:
        env_file: Path to .env file

    Returns:
        Validated TwinscribeConfig

    Raises:
        ConfigurationError: If config validation fails
        FileNotFoundError: If no config file found
    """
    global _global_loader, _global_config

    _global_loader = ConfigLoader(env_file=env_file)
    _global_config = _global_loader.load_from_env()
    return _global_config


def get_config() -> TwinscribeConfig:
    """Get the global configuration.

    Returns:
        Global TwinscribeConfig

    Raises:
        RuntimeError: If configuration not loaded
    """
    if _global_config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() or load_config_from_env() first."
        )
    return _global_config


def reload_config() -> TwinscribeConfig:
    """Reload the global configuration from its original path.

    Useful when the configuration file has been modified and you want
    to pick up the changes without restarting the application.

    Returns:
        Reloaded and validated TwinscribeConfig

    Raises:
        RuntimeError: If no configuration was previously loaded
        ConfigurationError: If reloaded config is invalid
    """
    global _global_config

    if _global_loader is None:
        raise RuntimeError(
            "Cannot reload: no configuration loaded. "
            "Call load_config() or load_config_from_env() first."
        )

    _global_config = _global_loader.reload()
    return _global_config


def reset_config() -> None:
    """Reset global configuration.

    Clears the cached configuration. Useful for testing.
    """
    global _global_loader, _global_config
    _global_loader = None
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

    return TwinscribeConfig(codebase=CodebaseConfig(path=codebase_path))


def get_loader() -> ConfigLoader | None:
    """Get the global config loader instance.

    Returns:
        The global ConfigLoader instance or None if not initialized.
    """
    return _global_loader
