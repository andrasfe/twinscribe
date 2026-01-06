"""
Environment Variable Handling.

Manages environment variables and secrets using python-dotenv.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


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

    openrouter_api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenRouter API key",
    )
    anthropic_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Direct Anthropic API key",
    )
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Direct OpenAI API key",
    )
    beads_api_token: Optional[SecretStr] = Field(
        default=None,
        description="Beads API token",
    )
    beads_username: Optional[str] = Field(
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

        Uses python-dotenv to load variables.

        Returns:
            True if loaded successfully
        """
        try:
            from dotenv import load_dotenv

            env_path = Path(self.env_file)
            if env_path.exists():
                load_dotenv(env_path)
                self.loaded = True
                return True
            else:
                self.errors.append(f".env file not found: {self.env_file}")
                return False
        except ImportError:
            self.errors.append("python-dotenv not installed")
            return False
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
_loader: Optional[EnvironmentLoader] = None
_config: Optional[EnvironmentConfig] = None


def load_environment(env_file: str = ".env") -> EnvironmentConfig:
    """Load environment configuration.

    Loads from .env file and caches the result.

    Args:
        env_file: Path to .env file

    Returns:
        EnvironmentConfig with loaded values
    """
    global _loader, _config

    if _loader is None or _loader.env_file != env_file:
        _loader = EnvironmentLoader(env_file=env_file)
        _loader.load()
        _config = _loader.get_config()

    return _config


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider.

    Args:
        provider: Provider name (openrouter, anthropic, openai)

    Returns:
        API key string or None if not set
    """
    config = load_environment()

    if provider == "openrouter":
        return (
            config.openrouter_api_key.get_secret_value()
            if config.openrouter_api_key
            else None
        )
    elif provider == "anthropic":
        return (
            config.anthropic_api_key.get_secret_value()
            if config.anthropic_api_key
            else None
        )
    elif provider == "openai":
        return (
            config.openai_api_key.get_secret_value()
            if config.openai_api_key
            else None
        )
    else:
        return None


def get_beads_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Beads credentials.

    Returns:
        Tuple of (username, api_token)
    """
    config = load_environment()

    username = config.beads_username
    token = (
        config.beads_api_token.get_secret_value()
        if config.beads_api_token
        else None
    )

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
    global _loader, _config
    _loader = None
    _config = None
