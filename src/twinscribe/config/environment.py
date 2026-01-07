"""
Environment Variable Handling.

Manages environment variables and secrets using python-dotenv.

IMPORTANT: Call ensure_dotenv_loaded() early in application startup
to ensure .env variables are available for Pydantic model defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr


# Track whether dotenv has been loaded
_dotenv_loaded: bool = False


def ensure_dotenv_loaded(env_file: str = ".env") -> bool:
    """Ensure .env file is loaded into os.environ.

    This function should be called EARLY in application startup,
    BEFORE any Pydantic models with environment-dependent defaults
    are instantiated.

    Args:
        env_file: Path to .env file (relative or absolute)

    Returns:
        True if .env was loaded successfully, False otherwise
    """
    global _dotenv_loaded

    if _dotenv_loaded:
        return True

    try:
        from dotenv import load_dotenv

        # Try multiple locations for .env
        env_paths = [
            Path(env_file),  # Relative to cwd
            Path.cwd() / env_file,  # Explicit cwd
            Path(__file__).parent.parent.parent.parent / env_file,  # Project root
        ]

        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                _dotenv_loaded = True
                return True

        # No .env file found, that's okay - use defaults
        _dotenv_loaded = True
        return False

    except ImportError:
        # python-dotenv not installed
        _dotenv_loaded = True
        return False


def get_env_model(
    env_var: str,
    default: str,
) -> str:
    """Get a model name from environment variable with fallback.

    This function is meant to be called at module load time
    to provide dynamic defaults for Pydantic model fields.

    Args:
        env_var: Environment variable name (e.g., "STREAM_A_DOCUMENTER_MODEL")
        default: Default value if env var is not set

    Returns:
        Model name from environment or default
    """
    # Ensure dotenv is loaded first
    ensure_dotenv_loaded()
    return os.environ.get(env_var, default)


class EnvironmentConfig(BaseModel):
    """Environment variables configuration.

    Manages API keys and other secrets from environment variables.
    Supports loading from .env files.

    Attributes:
        openrouter_api_key: OpenRouter API key (primary for all models)
        anthropic_api_key: Direct Anthropic API key (optional fallback)
        openai_api_key: Direct OpenAI API key (optional fallback)
        beads_api_token: Beads API token
        beads_username: Beads username (can also be in .env)
        env_file: Path to .env file
    """

    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Direct Anthropic API key",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Direct OpenAI API key",
    )
    beads_api_token: SecretStr | None = Field(
        default=None,
        description="Beads API token",
    )
    beads_username: str | None = Field(
        default=None,
        description="Beads username",
    )
    env_file: str = Field(
        default=".env",
        description="Path to .env file",
    )

    @property
    def has_openrouter(self) -> bool:
        """Check if OpenRouter is configured."""
        return self.openrouter_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return self.openai_api_key is not None

    @property
    def has_beads(self) -> bool:
        """Check if Beads is configured."""
        return self.beads_api_token is not None


# Environment variable names
ENV_VARS = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "beads_api_token": "BEADS_API_TOKEN",
    "beads_username": "BEADS_USERNAME",
}


@dataclass
class EnvironmentLoader:
    """Loads environment variables from .env files.

    Attributes:
        env_file: Path to .env file
        loaded: Whether .env has been loaded
        errors: Any errors encountered during loading
    """

    env_file: str = ".env"
    loaded: bool = False
    errors: list[str] = field(default_factory=list)

    def load(self) -> bool:
        """Load environment variables from .env file.

        Uses the global ensure_dotenv_loaded() function which respects
        the _dotenv_loaded flag to prevent double-loading in tests.

        Returns:
            True if loaded successfully
        """
        try:
            # Use ensure_dotenv_loaded() to respect the global flag
            # This prevents reloading .env during tests
            self.loaded = ensure_dotenv_loaded(self.env_file)
            if not self.loaded:
                env_path = Path(self.env_file)
                if not env_path.exists():
                    self.errors.append(f".env file not found: {self.env_file}")
            return self.loaded
        except Exception as e:
            self.errors.append(f"Error loading .env: {e}")
            return False

    def get_config(self) -> EnvironmentConfig:
        """Build EnvironmentConfig from loaded environment.

        Returns:
            EnvironmentConfig with values from environment
        """
        if not self.loaded:
            self.load()

        values = {}
        for config_key, env_var in ENV_VARS.items():
            value = os.environ.get(env_var)
            if value:
                if "api_key" in config_key or "token" in config_key:
                    values[config_key] = SecretStr(value)
                else:
                    values[config_key] = value

        values["env_file"] = self.env_file
        return EnvironmentConfig(**values)


# Global loader instance
_loader: EnvironmentLoader | None = None
_config: EnvironmentConfig | None = None


def load_environment(env_file: str = ".env") -> EnvironmentConfig:
    """Load environment configuration.

    Loads from .env file and caches the result.

    Args:
        env_file: Path to .env file

    Returns:
        EnvironmentConfig with loaded values
    """
    global _loader, _config

    # Ensure dotenv is loaded first (idempotent)
    ensure_dotenv_loaded(env_file)

    if _loader is None or _loader.env_file != env_file:
        _loader = EnvironmentLoader(env_file=env_file)
        _loader.load()
        _config = _loader.get_config()

    return _config


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider.

    Args:
        provider: Provider name (openrouter, anthropic, openai)

    Returns:
        API key string or None if not set
    """
    config = load_environment()

    if provider == "openrouter":
        return config.openrouter_api_key.get_secret_value() if config.openrouter_api_key else None
    elif provider == "anthropic":
        return config.anthropic_api_key.get_secret_value() if config.anthropic_api_key else None
    elif provider == "openai":
        return config.openai_api_key.get_secret_value() if config.openai_api_key else None
    else:
        return None


def get_beads_credentials() -> tuple[str | None, str | None]:
    """Get Beads credentials.

    Returns:
        Tuple of (username, api_token)
    """
    config = load_environment()

    username = config.beads_username
    token = config.beads_api_token.get_secret_value() if config.beads_api_token else None

    return username, token


def validate_environment(require_openrouter: bool = True) -> list[str]:
    """Validate that required environment variables are set.

    Args:
        require_openrouter: Whether OpenRouter is required

    Returns:
        List of missing/invalid variables
    """
    config = load_environment()
    missing = []

    if require_openrouter and not config.has_openrouter:
        missing.append("OPENROUTER_API_KEY")

    return missing


def reset_environment() -> None:
    """Reset cached environment configuration.

    Useful for testing or reloading after .env changes.
    """
    global _loader, _config, _dotenv_loaded
    _loader = None
    _config = None
    _dotenv_loaded = False
