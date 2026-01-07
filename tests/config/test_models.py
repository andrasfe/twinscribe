"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from twinscribe.config.models import (
    DEFAULT_MODELS,
    BeadsConfig,
    CodebaseConfig,
    ConvergenceConfig,
    Language,
    LoggingConfig,
    LogLevel,
    ModelConfig,
    ModelProvider,
    ModelsConfig,
    ModelTier,
    OutputConfig,
    StaticAnalysisConfig,
    StreamModelsConfig,
    TwinscribeConfig,
    VerificationConfig,
    VerificationStrategy,
)


class TestCodebaseConfig:
    """Tests for CodebaseConfig model."""

    def test_minimal_config(self):
        """Test creating config with only required fields."""
        config = CodebaseConfig(path="/my/codebase")

        assert config.path == "/my/codebase"
        assert config.language == Language.PYTHON
        assert len(config.exclude_patterns) > 0
        assert config.include_patterns == []
        assert config.entry_points == []

    def test_full_config(self):
        """Test creating config with all fields."""
        config = CodebaseConfig(
            path="/my/codebase",
            language=Language.JAVA,
            exclude_patterns=["**/test/**"],
            include_patterns=["**/*.java"],
            entry_points=["com.example.Main"],
        )

        assert config.path == "/my/codebase"
        assert config.language == Language.JAVA
        assert config.exclude_patterns == ["**/test/**"]
        assert config.include_patterns == ["**/*.java"]
        assert config.entry_points == ["com.example.Main"]

    def test_empty_path_rejected(self):
        """Test that empty path is rejected."""
        with pytest.raises(ValidationError):
            CodebaseConfig(path="")

    def test_whitespace_path_rejected(self):
        """Test that whitespace-only path is rejected."""
        with pytest.raises(ValidationError):
            CodebaseConfig(path="   ")


class TestModelConfig:
    """Tests for ModelConfig model."""

    def test_minimal_config(self):
        """Test creating config with required fields."""
        config = ModelConfig(
            name="my-model",
            tier=ModelTier.GENERATION,
        )

        assert config.name == "my-model"
        assert config.tier == ModelTier.GENERATION
        assert config.provider == ModelProvider.OPENROUTER
        assert config.temperature == 0.0
        assert config.max_tokens == 4096

    def test_full_config(self):
        """Test creating config with all fields."""
        config = ModelConfig(
            name="my-model",
            provider=ModelProvider.ANTHROPIC,
            tier=ModelTier.ARBITRATION,
            cost_per_million_input=10.0,
            cost_per_million_output=50.0,
            max_tokens=8192,
            temperature=0.5,
            context_window=200000,
        )

        assert config.provider == ModelProvider.ANTHROPIC
        assert config.cost_per_million_input == 10.0
        assert config.max_tokens == 8192
        assert config.temperature == 0.5

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid boundary values
        ModelConfig(name="test", tier=ModelTier.GENERATION, temperature=0.0)
        ModelConfig(name="test", tier=ModelTier.GENERATION, temperature=2.0)

        # Invalid values
        with pytest.raises(ValidationError):
            ModelConfig(name="test", tier=ModelTier.GENERATION, temperature=-0.1)

        with pytest.raises(ValidationError):
            ModelConfig(name="test", tier=ModelTier.GENERATION, temperature=2.1)


class TestModelsConfig:
    """Tests for ModelsConfig model."""

    def test_default_config(self):
        """Test default model configuration."""
        config = ModelsConfig()

        assert config.stream_a.documenter == "claude-3.5-sonnet"
        assert config.stream_a.validator == "claude-3.5-haiku"
        assert config.stream_b.documenter == "gpt-4o"
        assert config.stream_b.validator == "gpt-4o-mini"
        assert config.comparator == "claude-opus-4"

    def test_get_model_config_default(self):
        """Test getting default model config."""
        config = ModelsConfig()
        model = config.get_model_config("claude-sonnet-4-5")

        assert model.name == "claude-3-5-sonnet-20241022"
        assert model.tier == ModelTier.GENERATION

    def test_get_model_config_custom(self):
        """Test getting custom model config."""
        custom_model = ModelConfig(
            name="custom-model",
            tier=ModelTier.GENERATION,
            provider=ModelProvider.OPENAI,
        )
        config = ModelsConfig(
            custom_models={"my-custom": custom_model},
        )

        model = config.get_model_config("my-custom")
        assert model.name == "custom-model"
        assert model.provider == ModelProvider.OPENAI

    def test_get_model_config_custom_overrides_default(self):
        """Test that custom model overrides default with same name."""
        custom_model = ModelConfig(
            name="custom-override",
            tier=ModelTier.ARBITRATION,
        )
        config = ModelsConfig(
            custom_models={"claude-sonnet-4-5": custom_model},
        )

        model = config.get_model_config("claude-sonnet-4-5")
        assert model.name == "custom-override"

    def test_get_model_config_unknown(self):
        """Test getting unknown model raises KeyError."""
        config = ModelsConfig()

        with pytest.raises(KeyError):
            config.get_model_config("nonexistent-model")


class TestConvergenceConfig:
    """Tests for ConvergenceConfig model."""

    def test_default_values(self):
        """Test default convergence values."""
        config = ConvergenceConfig()

        assert config.max_iterations == 5
        assert config.call_graph_match_threshold == 0.98
        assert config.documentation_similarity_threshold == 0.95
        assert config.max_open_discrepancies == 2

    def test_max_iterations_bounds(self):
        """Test max_iterations validation bounds."""
        ConvergenceConfig(max_iterations=1)
        ConvergenceConfig(max_iterations=20)

        with pytest.raises(ValidationError):
            ConvergenceConfig(max_iterations=0)

        with pytest.raises(ValidationError):
            ConvergenceConfig(max_iterations=21)

    def test_threshold_bounds(self):
        """Test threshold validation bounds."""
        ConvergenceConfig(call_graph_match_threshold=0.0)
        ConvergenceConfig(call_graph_match_threshold=1.0)

        with pytest.raises(ValidationError):
            ConvergenceConfig(call_graph_match_threshold=-0.1)

        with pytest.raises(ValidationError):
            ConvergenceConfig(call_graph_match_threshold=1.1)


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_default_values(self):
        """Test default logging values."""
        config = LoggingConfig()

        assert config.level == LogLevel.INFO
        assert config.file is None
        assert config.json_output is False
        assert config.include_timestamp is True
        assert config.include_module is True

    def test_all_log_levels(self):
        """Test all log levels are valid."""
        for level in LogLevel:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_get_format_string_default(self):
        """Test default format string generation."""
        config = LoggingConfig()
        fmt = config.get_format_string()

        assert "%(asctime)s" in fmt
        assert "%(name)s" in fmt
        assert "%(levelname)s" in fmt
        assert "%(message)s" in fmt

    def test_get_format_string_no_timestamp(self):
        """Test format string without timestamp."""
        config = LoggingConfig(include_timestamp=False)
        fmt = config.get_format_string()

        assert "%(asctime)s" not in fmt
        assert "%(levelname)s" in fmt

    def test_get_format_string_no_module(self):
        """Test format string without module name."""
        config = LoggingConfig(include_module=False)
        fmt = config.get_format_string()

        assert "%(name)s" not in fmt
        assert "%(levelname)s" in fmt

    def test_get_format_string_json(self):
        """Test JSON format string."""
        config = LoggingConfig(json_output=True)
        fmt = config.get_format_string()

        # JSON output should return simple format
        assert fmt == "%(message)s"


class TestOutputConfig:
    """Tests for OutputConfig model."""

    def test_default_values(self):
        """Test default output values."""
        config = OutputConfig()

        assert config.base_dir == "./output"
        assert config.documentation_file == "documentation.json"
        assert config.create_dirs is True

    def test_get_path(self):
        """Test get_path method."""
        config = OutputConfig(base_dir="/my/output")
        path = config.get_path("documentation_file")

        assert str(path) == "/my/output/documentation.json"


class TestTwinscribeConfig:
    """Tests for TwinscribeConfig root model."""

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = TwinscribeConfig(
            codebase=CodebaseConfig(path="/my/codebase"),
        )

        assert config.codebase.path == "/my/codebase"
        assert config.debug is False
        assert config.dry_run is False

    def test_full_config(self):
        """Test creating config with all sections."""
        config = TwinscribeConfig(
            codebase=CodebaseConfig(path="/my/codebase"),
            models=ModelsConfig(),
            convergence=ConvergenceConfig(max_iterations=10),
            beads=BeadsConfig(enabled=False),
            static_analysis=StaticAnalysisConfig(),
            verification=VerificationConfig(enabled=True),
            output=OutputConfig(base_dir="/custom/output"),
            logging=LoggingConfig(level=LogLevel.DEBUG),
            debug=True,
            dry_run=True,
        )

        assert config.convergence.max_iterations == 10
        assert config.beads.enabled is False
        assert config.verification.enabled is True
        assert config.output.base_dir == "/custom/output"
        assert config.logging.level == LogLevel.DEBUG
        assert config.debug is True
        assert config.dry_run is True

    def test_model_validation(self):
        """Test that model references are validated."""
        # Valid configuration with default models
        TwinscribeConfig(
            codebase=CodebaseConfig(path="/my/codebase"),
        )

        # Invalid configuration with unknown model raises KeyError
        # (from model_validator which calls get_model_config)
        with pytest.raises(KeyError, match="Unknown model"):
            TwinscribeConfig(
                codebase=CodebaseConfig(path="/my/codebase"),
                models=ModelsConfig(
                    stream_a=StreamModelsConfig(
                        documenter="nonexistent-model",
                        validator="claude-haiku-4-5",
                    ),
                ),
            )

    def test_to_yaml_dict(self):
        """Test to_yaml_dict method."""
        config = TwinscribeConfig(
            codebase=CodebaseConfig(path="/my/codebase"),
        )
        yaml_dict = config.to_yaml_dict()

        assert isinstance(yaml_dict, dict)
        assert yaml_dict["codebase"]["path"] == "/my/codebase"
        assert "debug" in yaml_dict


class TestVerificationConfig:
    """Tests for VerificationConfig model."""

    def test_default_disabled(self):
        """Test verification is disabled by default."""
        config = VerificationConfig()
        assert config.enabled is False

    def test_all_strategies_enabled_by_default(self):
        """Test all strategies are in enabled_strategies by default."""
        config = VerificationConfig()

        for strategy in VerificationStrategy:
            assert strategy in config.enabled_strategies

    def test_is_strategy_enabled(self):
        """Test is_strategy_enabled method."""
        config = VerificationConfig(
            enabled=True,
            enabled_strategies=[VerificationStrategy.QA_INTERROGATION],
        )

        assert config.is_strategy_enabled(VerificationStrategy.QA_INTERROGATION) is True
        assert config.is_strategy_enabled(VerificationStrategy.MUTATION_DETECTION) is False

    def test_is_strategy_enabled_when_disabled(self):
        """Test is_strategy_enabled returns False when verification is disabled."""
        config = VerificationConfig(
            enabled=False,
            enabled_strategies=[VerificationStrategy.QA_INTERROGATION],
        )

        # Even though strategy is in list, verification is disabled
        assert config.is_strategy_enabled(VerificationStrategy.QA_INTERROGATION) is False


class TestDefaultModels:
    """Tests for DEFAULT_MODELS configuration."""

    def test_all_tiers_present(self):
        """Test that all tiers have at least one model."""
        tiers_present = {model.tier for model in DEFAULT_MODELS.values()}

        assert ModelTier.GENERATION in tiers_present
        assert ModelTier.VALIDATION in tiers_present
        assert ModelTier.ARBITRATION in tiers_present

    def test_expected_models(self):
        """Test expected models are present."""
        expected = [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-opus-4-5",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        for model_name in expected:
            assert model_name in DEFAULT_MODELS, f"Expected {model_name} in DEFAULT_MODELS"

    def test_model_costs_positive(self):
        """Test all model costs are non-negative."""
        for name, model in DEFAULT_MODELS.items():
            assert model.cost_per_million_input >= 0, f"{name} has negative input cost"
            assert model.cost_per_million_output >= 0, f"{name} has negative output cost"
