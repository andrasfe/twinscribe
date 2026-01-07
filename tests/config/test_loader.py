"""Tests for configuration loader."""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from twinscribe.config import (
    CONFIG_ENV_VAR,
    ENV_VAR_OVERRIDES,
    ConfigLoader,
    ConfigurationError,
    LogLevel,
    TwinscribeConfig,
    create_default_config,
    get_config,
    load_config,
    load_config_from_env,
    reload_config,
    reset_config,
    reset_environment,
)


# List of ALL environment variables that may affect config tests
# This includes both .env vars and TWINSCRIBE_* overrides
# IMPORTANT: ALL vars in ENV_VAR_OVERRIDES must be cleared to prevent
# .env values from overriding test-specific values
_ENV_VARS_FROM_DOTENV = [
    # Model configuration
    "STREAM_A_DOCUMENTER_MODEL",
    "STREAM_A_VALIDATOR_MODEL",
    "STREAM_B_DOCUMENTER_MODEL",
    "STREAM_B_VALIDATOR_MODEL",
    "COMPARATOR_MODEL",
    "TWINSCRIBE_MODEL_PROVIDER",
    "TWINSCRIBE_STREAM_A_DOCUMENTER",
    "TWINSCRIBE_STREAM_A_VALIDATOR",
    "TWINSCRIBE_STREAM_B_DOCUMENTER",
    "TWINSCRIBE_STREAM_B_VALIDATOR",
    "TWINSCRIBE_COMPARATOR",
    # Convergence settings
    "MAX_ITERATIONS",
    "CALL_GRAPH_MATCH_THRESHOLD",
    "DOCUMENTATION_SIMILARITY_THRESHOLD",
    "TWINSCRIBE_MAX_ITERATIONS",
    "TWINSCRIBE_CONVERGENCE_THRESHOLD",
    # Output settings
    "OUTPUT_DIR",
    "TWINSCRIBE_OUTPUT_DIR",
    # Logging settings
    "LOG_LEVEL",
    "TWINSCRIBE_LOG_LEVEL",
    "TWINSCRIBE_LOG_FILE",
    # Debug/runtime flags
    "TWINSCRIBE_DEBUG",
    "TWINSCRIBE_DRY_RUN",
]


@pytest.fixture(autouse=True)
def reset_global_config(monkeypatch):
    """Reset global config and environment before and after each test.

    This ensures tests are isolated from:
    1. Previous test state
    2. Values loaded from .env file
    """
    import twinscribe.config.environment as env_module
    import twinscribe.config.models as models_module

    # Reset environment state
    reset_config()
    reset_environment()

    # Remove ALL env vars that could have been loaded from .env
    # Using monkeypatch ensures they're restored after each test
    for var in _ENV_VARS_FROM_DOTENV:
        monkeypatch.delenv(var, raising=False)

    # CRITICAL: Prevent ensure_dotenv_loaded() from reloading .env during tests
    # by marking it as already loaded (tests will set their own env vars)
    monkeypatch.setattr(env_module, "_dotenv_loaded", True)

    yield

    # Clean up after test
    reset_config()
    reset_environment()


@pytest.fixture
def minimal_config_dict():
    """Minimal valid configuration dictionary."""
    return {
        "codebase": {
            "path": "/tmp/test/codebase",
        }
    }


@pytest.fixture
def full_config_dict():
    """Full configuration dictionary with all sections."""
    return {
        "codebase": {
            "path": "/tmp/test/codebase",
            "language": "python",
            "exclude_patterns": ["**/test_*", "**/__pycache__/**"],
            "include_patterns": [],
            "entry_points": ["main"],
        },
        "models": {
            "stream_a": {
                "documenter": "claude-sonnet-4-5",
                "validator": "claude-haiku-4-5",
            },
            "stream_b": {
                "documenter": "gpt-4o",
                "validator": "gpt-4o-mini",
            },
            "comparator": "claude-opus-4-5",
        },
        "convergence": {
            "max_iterations": 5,
            "call_graph_match_threshold": 0.98,
            "documentation_similarity_threshold": 0.95,
            "max_open_discrepancies": 2,
        },
        "output": {
            "base_dir": "./output",
            "documentation_file": "documentation.json",
        },
        "logging": {
            "level": "INFO",
            "json_output": False,
        },
        "debug": False,
        "dry_run": False,
    }


@pytest.fixture
def config_file(minimal_config_dict):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    ) as f:
        yaml.safe_dump(minimal_config_dict, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture
def full_config_file(full_config_dict):
    """Create a temporary config file with full configuration."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    ) as f:
        yaml.safe_dump(full_config_dict, f)
        f.flush()
        yield Path(f.name)
    os.unlink(f.name)


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_minimal_config(self, config_file):
        """Test loading minimal configuration."""
        loader = ConfigLoader(config_file)
        config = loader.load()

        assert isinstance(config, TwinscribeConfig)
        assert config.codebase.path == "/tmp/test/codebase"
        # Check defaults are applied
        assert config.convergence.max_iterations == 5
        assert config.debug is False

    def test_load_full_config(self, full_config_file):
        """Test loading full configuration."""
        loader = ConfigLoader(full_config_file)
        config = loader.load()

        assert config.codebase.path == "/tmp/test/codebase"
        assert config.models.stream_a.documenter == "claude-sonnet-4-5"
        assert config.convergence.max_iterations == 5
        assert config.logging.level == LogLevel.INFO
        assert config.logging.json_output is False

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        loader = ConfigLoader("/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            f.write("invalid: yaml: content: ][")
            f.flush()
            path = Path(f.name)

        try:
            loader = ConfigLoader(path)
            with pytest.raises(ConfigurationError):
                loader.load()
        finally:
            os.unlink(path)

    def test_load_missing_required_field(self):
        """Test loading config without required field raises error."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            # codebase.path is required
            yaml.safe_dump({"debug": True}, f)
            f.flush()
            path = Path(f.name)

        try:
            loader = ConfigLoader(path)
            with pytest.raises(ConfigurationError) as exc_info:
                loader.load()
            assert "codebase" in str(exc_info.value).lower()
        finally:
            os.unlink(path)

    def test_get_before_load(self):
        """Test get() before load() raises error."""
        loader = ConfigLoader()

        with pytest.raises(RuntimeError) as exc_info:
            loader.get()
        assert "not loaded" in str(exc_info.value).lower()

    def test_get_after_load(self, config_file):
        """Test get() returns config after load()."""
        loader = ConfigLoader(config_file)
        config1 = loader.load()
        config2 = loader.get()

        assert config1 is config2

    def test_reload_config(self, config_file, minimal_config_dict):
        """Test reloading configuration."""
        loader = ConfigLoader(config_file)
        config1 = loader.load()

        # Modify the file
        minimal_config_dict["debug"] = True
        with open(config_file, "w") as f:
            yaml.safe_dump(minimal_config_dict, f)

        config2 = loader.reload()

        assert config2.debug is True
        assert config1 is not config2

    def test_reload_before_load(self):
        """Test reload() before load() raises error."""
        loader = ConfigLoader()

        with pytest.raises(RuntimeError):
            loader.reload()


class TestEnvironmentVariableSubstitution:
    """Tests for environment variable substitution in config."""

    def test_env_var_substitution_simple(self, minimal_config_dict):
        """Test simple environment variable substitution."""
        minimal_config_dict["debug"] = "${TEST_DEBUG}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch.dict(os.environ, {"TEST_DEBUG": "true"}):
                loader = ConfigLoader(path)
                config = loader.load()
                assert config.debug is True
        finally:
            os.unlink(path)

    def test_env_var_with_default(self, minimal_config_dict):
        """Test environment variable with default value."""
        minimal_config_dict["debug"] = "${NONEXISTENT_VAR:-false}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            # Ensure env var is not set
            env = os.environ.copy()
            env.pop("NONEXISTENT_VAR", None)

            with mock.patch.dict(os.environ, env, clear=True):
                loader = ConfigLoader(path)
                config = loader.load()
                assert config.debug is False
        finally:
            os.unlink(path)

    def test_env_var_with_default_colon_syntax(self, minimal_config_dict):
        """Test environment variable with default (colon syntax)."""
        minimal_config_dict["debug"] = "${NONEXISTENT_VAR:false}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            loader = ConfigLoader(path)
            config = loader.load()
            assert config.debug is False
        finally:
            os.unlink(path)

    def test_env_var_boolean_coercion(self, minimal_config_dict):
        """Test boolean coercion from environment variables."""
        minimal_config_dict["debug"] = "${TEST_BOOL}"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            # Test various boolean representations
            for value, expected in [
                ("true", True),
                ("TRUE", True),
                ("True", True),
                ("yes", True),
                ("on", True),
                ("1", True),
                ("false", False),
                ("FALSE", False),
                ("no", False),
                ("off", False),
                ("0", False),
            ]:
                with mock.patch.dict(os.environ, {"TEST_BOOL": value}):
                    loader = ConfigLoader(path)
                    config = loader.load()
                    assert config.debug is expected, f"Failed for '{value}'"
        finally:
            os.unlink(path)

    def test_env_var_numeric_coercion(self, minimal_config_dict):
        """Test numeric coercion from environment variables."""
        minimal_config_dict["convergence"] = {
            "max_iterations": "${TEST_INT}",
            "call_graph_match_threshold": "${TEST_FLOAT}",
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch.dict(
                os.environ,
                {"TEST_INT": "10", "TEST_FLOAT": "0.85"},
            ):
                loader = ConfigLoader(path)
                config = loader.load()
                assert config.convergence.max_iterations == 10
                assert config.convergence.call_graph_match_threshold == 0.85
        finally:
            os.unlink(path)


class TestEnvironmentVariableOverrides:
    """Tests for TWINSCRIBE_* environment variable overrides."""

    def test_override_debug(self, config_file):
        """Test TWINSCRIBE_DEBUG override."""
        with mock.patch.dict(os.environ, {"TWINSCRIBE_DEBUG": "true"}):
            loader = ConfigLoader(config_file)
            config = loader.load()
            assert config.debug is True

    def test_override_dry_run(self, config_file):
        """Test TWINSCRIBE_DRY_RUN override."""
        with mock.patch.dict(os.environ, {"TWINSCRIBE_DRY_RUN": "true"}):
            loader = ConfigLoader(config_file)
            config = loader.load()
            assert config.dry_run is True

    def test_override_log_level(self, config_file):
        """Test TWINSCRIBE_LOG_LEVEL override."""
        with mock.patch.dict(os.environ, {"TWINSCRIBE_LOG_LEVEL": "DEBUG"}):
            loader = ConfigLoader(config_file)
            config = loader.load()
            assert config.logging.level == LogLevel.DEBUG

    def test_override_output_dir(self, config_file):
        """Test TWINSCRIBE_OUTPUT_DIR override."""
        with mock.patch.dict(os.environ, {"TWINSCRIBE_OUTPUT_DIR": "/custom/output"}):
            loader = ConfigLoader(config_file)
            config = loader.load()
            assert config.output.base_dir == "/custom/output"

    def test_override_max_iterations(self, config_file):
        """Test TWINSCRIBE_MAX_ITERATIONS override."""
        with mock.patch.dict(os.environ, {"TWINSCRIBE_MAX_ITERATIONS": "10"}):
            loader = ConfigLoader(config_file)
            config = loader.load()
            assert config.convergence.max_iterations == 10

    def test_override_stream_a_documenter(self, full_config_file):
        """Test TWINSCRIBE_STREAM_A_DOCUMENTER override."""
        # Use full config file which has stream_a.validator already set
        with mock.patch.dict(
            os.environ,
            {"TWINSCRIBE_STREAM_A_DOCUMENTER": "gpt-4o"},
        ):
            loader = ConfigLoader(full_config_file)
            config = loader.load()
            assert config.models.stream_a.documenter == "gpt-4o"

    def test_override_precedence(self, minimal_config_dict):
        """Test that env var overrides take precedence over YAML."""
        minimal_config_dict["debug"] = False

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.safe_dump(minimal_config_dict, f)
            f.flush()
            path = Path(f.name)

        try:
            with mock.patch.dict(os.environ, {"TWINSCRIBE_DEBUG": "true"}):
                loader = ConfigLoader(path)
                config = loader.load()
                assert config.debug is True
        finally:
            os.unlink(path)

    def test_documented_env_vars(self):
        """Test that ENV_VAR_OVERRIDES contains expected vars."""
        expected_vars = [
            "TWINSCRIBE_DEBUG",
            "TWINSCRIBE_DRY_RUN",
            "TWINSCRIBE_LOG_LEVEL",
            "TWINSCRIBE_OUTPUT_DIR",
            "TWINSCRIBE_MAX_ITERATIONS",
        ]
        for var in expected_vars:
            assert var in ENV_VAR_OVERRIDES, f"Expected {var} in ENV_VAR_OVERRIDES"


class TestGlobalConfigFunctions:
    """Tests for global configuration functions."""

    def test_load_config(self, config_file):
        """Test load_config function."""
        config = load_config(config_file)

        assert isinstance(config, TwinscribeConfig)
        assert config.codebase.path == "/tmp/test/codebase"

    def test_get_config_after_load(self, config_file):
        """Test get_config after load_config."""
        config1 = load_config(config_file)
        config2 = get_config()

        assert config1 is config2

    def test_get_config_before_load(self):
        """Test get_config before loading raises error."""
        with pytest.raises(RuntimeError):
            get_config()

    def test_reload_config_function(self, config_file, minimal_config_dict):
        """Test reload_config function."""
        load_config(config_file)

        # Modify the file
        minimal_config_dict["debug"] = True
        with open(config_file, "w") as f:
            yaml.safe_dump(minimal_config_dict, f)

        config = reload_config()
        assert config.debug is True

    def test_create_default_config(self):
        """Test create_default_config function."""
        config = create_default_config("/my/codebase")

        assert config.codebase.path == "/my/codebase"
        assert config.convergence.max_iterations == 5
        assert config.logging.level == LogLevel.INFO

    def test_load_config_from_env_with_env_var(self, config_file):
        """Test load_config_from_env with TWINSCRIBE_CONFIG."""
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(config_file)}):
            config = load_config_from_env()
            assert config.codebase.path == "/tmp/test/codebase"


class TestConfigLoaderFromEnv:
    """Tests for load_from_env functionality."""

    def test_load_from_env_with_config_env_var(self, config_file):
        """Test loading from TWINSCRIBE_CONFIG environment variable."""
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: str(config_file)}):
            loader = ConfigLoader()
            config = loader.load_from_env()
            assert config.codebase.path == "/tmp/test/codebase"

    def test_load_from_env_invalid_path(self):
        """Test loading from invalid path in env var raises error."""
        with mock.patch.dict(os.environ, {CONFIG_ENV_VAR: "/nonexistent.yaml"}):
            loader = ConfigLoader()
            with pytest.raises(FileNotFoundError):
                loader.load_from_env()

    def test_load_from_env_default_locations(self, minimal_config_dict):
        """Test loading from default locations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            with open(config_path, "w") as f:
                yaml.safe_dump(minimal_config_dict, f)

            # Change to temp directory and remove TWINSCRIBE_CONFIG
            original_cwd = os.getcwd()
            env = os.environ.copy()
            env.pop(CONFIG_ENV_VAR, None)

            try:
                os.chdir(tmpdir)
                with mock.patch.dict(os.environ, env, clear=True):
                    loader = ConfigLoader()
                    config = loader.load_from_env()
                    assert config.codebase.path == "/tmp/test/codebase"
            finally:
                os.chdir(original_cwd)

    def test_load_from_env_no_config_found(self):
        """Test load_from_env with no config file raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            env = os.environ.copy()
            env.pop(CONFIG_ENV_VAR, None)

            try:
                os.chdir(tmpdir)
                with mock.patch.dict(os.environ, env, clear=True):
                    loader = ConfigLoader()
                    with pytest.raises(FileNotFoundError) as exc_info:
                        loader.load_from_env()
                    assert "No configuration file found" in str(exc_info.value)
            finally:
                os.chdir(original_cwd)


class TestConfigSave:
    """Tests for configuration save functionality."""

    def test_save_config(self, config_file):
        """Test saving configuration to file."""
        loader = ConfigLoader(config_file)
        config = loader.load()

        # Save to new file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            save_path = Path(f.name)

        try:
            loader.save(save_path)

            # Load saved config and verify
            loader2 = ConfigLoader(save_path)
            config2 = loader2.load()

            assert config.codebase.path == config2.codebase.path
            assert config.debug == config2.debug
        finally:
            os.unlink(save_path)

    def test_save_without_load(self):
        """Test save without loading raises error."""
        loader = ConfigLoader()

        with pytest.raises(ValueError):
            loader.save("/tmp/config.yaml")

    def test_save_without_path(self, config_file):
        """Test save without path uses original path."""
        loader = ConfigLoader(config_file)
        loader.load()

        # Modify config and save back
        loader._config = loader._config.model_copy(update={"debug": True})
        loader.save()

        # Reload and verify
        loader2 = ConfigLoader(config_file)
        config2 = loader2.load()
        assert config2.debug is True
