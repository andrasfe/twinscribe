"""
Configuration Data Models.

Defines all configuration schemas using Pydantic for validation
and type safety.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"


class ModelTier(str, Enum):
    """Model cost tier."""

    GENERATION = "generation"
    VALIDATION = "validation"
    ARBITRATION = "arbitration"


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    MULTI = "multi"


class AnalyzerType(str, Enum):
    """Static analyzer types."""

    PYCG = "pycg"
    PYAN3 = "pyan3"
    JAVA_CALLGRAPH = "java-callgraph"
    WALA = "wala"
    TS_CALLGRAPH = "typescript-call-graph"
    SOURCETRAIL = "sourcetrail"


class CodebaseConfig(BaseModel):
    """Configuration for the codebase to document.

    Attributes:
        path: Path to the codebase root
        language: Primary programming language
        exclude_patterns: Glob patterns to exclude
        include_patterns: Glob patterns to include (optional)
        entry_points: Entry point modules for analysis
    """

    path: str = Field(
        ...,
        description="Path to codebase root",
        examples=["/path/to/legacy/codebase", "./src"],
    )
    language: Language = Field(
        default=Language.PYTHON,
        description="Primary language",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "**/test_*",
            "**/tests/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/node_modules/**",
        ],
        description="Patterns to exclude",
    )
    include_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns to include (empty = all)",
    )
    entry_points: list[str] = Field(
        default_factory=list,
        description="Entry point modules",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate that path is not empty."""
        if not v.strip():
            raise ValueError("Codebase path cannot be empty")
        return v


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Attributes:
        name: Model identifier
        provider: Model provider
        tier: Cost tier
        cost_per_million_input: Input token cost
        cost_per_million_output: Output token cost
        max_tokens: Maximum output tokens
        temperature: Sampling temperature
        context_window: Context window size
    """

    name: str = Field(
        ...,
        description="Model identifier",
        examples=["claude-sonnet-4-5-20250929", "gpt-4o"],
    )
    provider: ModelProvider = Field(
        default=ModelProvider.OPENROUTER,
        description="Model provider",
    )
    tier: ModelTier = Field(
        ...,
        description="Cost tier",
    )
    cost_per_million_input: float = Field(
        default=0.0,
        ge=0.0,
        description="Input cost per million tokens",
    )
    cost_per_million_output: float = Field(
        default=0.0,
        ge=0.0,
        description="Output cost per million tokens",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Max output tokens",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    context_window: int = Field(
        default=128000,
        ge=1,
        description="Context window size",
    )


# Default model configurations
# Model names must be valid OpenRouter model IDs (without provider prefix)
_SONNET_CONFIG = ModelConfig(
    name="claude-3-5-sonnet-20241022",
    provider=ModelProvider.OPENROUTER,
    tier=ModelTier.GENERATION,
    cost_per_million_input=3.0,
    cost_per_million_output=15.0,
    max_tokens=4096,
    context_window=200000,
)
_HAIKU_CONFIG = ModelConfig(
    name="claude-3-5-haiku-20241022",
    provider=ModelProvider.OPENROUTER,
    tier=ModelTier.VALIDATION,
    cost_per_million_input=0.80,
    cost_per_million_output=4.0,
    max_tokens=2048,
    context_window=200000,
)
_OPUS_CONFIG = ModelConfig(
    name="claude-opus-4-20250514",
    provider=ModelProvider.OPENROUTER,
    tier=ModelTier.ARBITRATION,
    cost_per_million_input=15.0,
    cost_per_million_output=75.0,
    max_tokens=8192,
    context_window=200000,
)

DEFAULT_MODELS = {
    # Claude models with various aliases
    "claude-sonnet-4": _SONNET_CONFIG,
    "claude-sonnet-4-5": _SONNET_CONFIG,  # Alias for tests
    "claude-3.5-sonnet": _SONNET_CONFIG,
    "claude-haiku-4-5": _HAIKU_CONFIG,  # Alias for tests
    "claude-3.5-haiku": _HAIKU_CONFIG,
    "claude-opus-4": _OPUS_CONFIG,
    "claude-opus-4-5": _OPUS_CONFIG,  # Alias for tests
    # GPT models
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.GENERATION,
        cost_per_million_input=2.5,
        cost_per_million_output=10.0,
        max_tokens=4096,
        context_window=128000,
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.VALIDATION,
        cost_per_million_input=0.15,
        cost_per_million_output=0.60,
        max_tokens=2048,
        context_window=128000,
    ),
}


class StreamModelsConfig(BaseModel):
    """Model configuration for one stream.

    Attributes:
        documenter: Model for documentation generation
        validator: Model for validation
    """

    documenter: str = Field(
        ...,
        description="Documenter model name",
    )
    validator: str = Field(
        ...,
        description="Validator model name",
    )


class ModelsConfig(BaseModel):
    """Model configuration for all agents.

    Attributes:
        stream_a: Stream A model configuration
        stream_b: Stream B model configuration
        comparator: Comparator model name
        custom_models: Custom model definitions
    """

    stream_a: StreamModelsConfig = Field(
        default_factory=lambda: StreamModelsConfig(
            documenter="claude-3.5-sonnet",
            validator="claude-3.5-haiku",
        ),
        description="Stream A models",
    )
    stream_b: StreamModelsConfig = Field(
        default_factory=lambda: StreamModelsConfig(
            documenter="gpt-4o",
            validator="gpt-4o-mini",
        ),
        description="Stream B models",
    )
    comparator: str = Field(
        default="claude-opus-4",
        description="Comparator model",
    )
    custom_models: dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Custom model definitions",
    )

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a model by name.

        Args:
            model_name: Model name or alias

        Returns:
            Model configuration

        Raises:
            KeyError: If model not found
        """
        # Check custom models first
        if model_name in self.custom_models:
            return self.custom_models[model_name]
        # Then check defaults
        if model_name in DEFAULT_MODELS:
            return DEFAULT_MODELS[model_name]
        raise KeyError(f"Unknown model: {model_name}")


class ConvergenceConfig(BaseModel):
    """Configuration for convergence criteria.

    Attributes:
        max_iterations: Maximum iterations before forced convergence
        call_graph_match_threshold: Required call graph match rate
        documentation_similarity_threshold: Required doc similarity
        max_open_discrepancies: Max unresolved non-blocking issues
        beads_ticket_timeout_hours: Timeout waiting for Beads resolution
    """

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum iterations",
    )
    call_graph_match_threshold: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Required call graph match rate",
    )
    documentation_similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Required documentation similarity",
    )
    max_open_discrepancies: int = Field(
        default=2,
        ge=0,
        description="Max unresolved discrepancies allowed",
    )
    beads_ticket_timeout_hours: int = Field(
        default=48,
        ge=0,
        description="Timeout for Beads ticket resolution (0 = no wait)",
    )


class BeadsConfig(BaseModel):
    """Configuration for Beads integration.

    Attributes:
        enabled: Whether Beads integration is enabled
        server: Beads server URL
        project: Project key for discrepancy tickets
        rebuild_project: Project key for rebuild tickets
        username: Beads username (can use env var)
        api_token_env: Environment variable for API token
        ticket_labels: Default labels for tickets
        auto_create_tickets: Automatically create tickets
    """

    enabled: bool = Field(
        default=True,
        description="Enable Beads integration",
    )
    server: str = Field(
        default="",
        description="Beads server URL",
        examples=["https://your-org.atlassian.net"],
    )
    project: str = Field(
        default="LEGACY_DOC",
        description="Discrepancy ticket project",
    )
    rebuild_project: str = Field(
        default="REBUILD",
        description="Rebuild ticket project",
    )
    username: str = Field(
        default="",
        description="Beads username",
    )
    api_token_env: str = Field(
        default="BEADS_API_TOKEN",
        description="Env var for API token",
    )
    ticket_labels: list[str] = Field(
        default_factory=lambda: ["ai-documentation"],
        description="Default ticket labels",
    )
    auto_create_tickets: bool = Field(
        default=True,
        description="Auto-create tickets for discrepancies",
    )


class AnalyzerToolConfig(BaseModel):
    """Configuration for a specific analyzer tool.

    Attributes:
        enabled: Whether this analyzer is enabled
        executable: Path to executable (optional)
        extra_args: Additional command-line arguments
        timeout_seconds: Execution timeout
    """

    enabled: bool = Field(default=True)
    executable: str | None = Field(default=None)
    extra_args: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=300, ge=1)


class StaticAnalysisConfig(BaseModel):
    """Configuration for static analysis.

    Attributes:
        python: Python analyzer config
        java: Java analyzer config
        javascript: JavaScript analyzer config
        multi_language_fallback: Multi-language fallback analyzer
        cache_enabled: Enable result caching
        cache_ttl_hours: Cache time-to-live
    """

    python: dict[str, AnalyzerToolConfig] = Field(
        default_factory=lambda: {
            "primary": AnalyzerToolConfig(enabled=True),
            "fallback": AnalyzerToolConfig(enabled=True),
        },
    )
    java: dict[str, AnalyzerToolConfig] = Field(
        default_factory=lambda: {
            "primary": AnalyzerToolConfig(enabled=True),
            "fallback": AnalyzerToolConfig(enabled=True),
        },
    )
    javascript: dict[str, AnalyzerToolConfig] = Field(
        default_factory=lambda: {
            "primary": AnalyzerToolConfig(enabled=True),
        },
    )
    multi_language_fallback: AnalyzerToolConfig = Field(
        default_factory=lambda: AnalyzerToolConfig(
            enabled=True,
            timeout_seconds=600,
        ),
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable result caching",
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=0,
        description="Cache TTL (0 = no expiry)",
    )


class VerificationStrategy(str, Enum):
    """Available verification strategies."""

    QA_INTERROGATION = "qa_interrogation"
    MASKED_RECONSTRUCTION = "masked_reconstruction"
    SCENARIO_WALKTHROUGH = "scenario_walkthrough"
    MUTATION_DETECTION = "mutation_detection"
    IMPACT_ANALYSIS = "impact_analysis"
    ADVERSARIAL_REVIEW = "adversarial_review"
    TEST_GENERATION = "test_generation"


class VerificationThresholdsConfig(BaseModel):
    """Threshold configuration for verification quality scores.

    Attributes:
        min_overall_quality: Minimum overall quality score to pass verification
        min_qa_score: Minimum Q&A interrogation score
        min_reconstruction_score: Minimum masked reconstruction score
        min_scenario_score: Minimum scenario walkthrough score
        min_test_pass_rate: Minimum generated test pass rate
    """

    min_overall_quality: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum overall quality score to pass",
    )
    min_qa_score: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum Q&A interrogation score",
    )
    min_reconstruction_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum masked reconstruction score",
    )
    min_scenario_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum scenario walkthrough score",
    )
    min_test_pass_rate: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum generated test pass rate",
    )


class QAQuestionCategory(str, Enum):
    """Categories for Q&A interrogation questions."""

    RETURN_VALUE = "return_value"
    ERROR_HANDLING = "error_handling"
    EDGE_CASE = "edge_case"
    CALL_FLOW = "call_flow"


class QAInterrogationConfig(BaseModel):
    """Configuration for Q&A interrogation verification strategy.

    This strategy generates questions about code behavior and validates
    that the documentation can answer them correctly.

    Attributes:
        questions_per_component: Number of questions to generate per component
        categories: Question categories to include
    """

    questions_per_component: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of questions per component",
    )
    categories: list[QAQuestionCategory] = Field(
        default_factory=lambda: [
            QAQuestionCategory.RETURN_VALUE,
            QAQuestionCategory.ERROR_HANDLING,
            QAQuestionCategory.EDGE_CASE,
            QAQuestionCategory.CALL_FLOW,
        ],
        description="Question categories to include",
    )


class MaskType(str, Enum):
    """Types of code elements that can be masked for reconstruction."""

    CONSTANTS = "constants"
    CONDITIONS = "conditions"
    RETURNS = "returns"


class MaskedReconstructionConfig(BaseModel):
    """Configuration for masked reconstruction verification strategy.

    This strategy masks portions of code and validates that documentation
    contains enough information to reconstruct the masked elements.

    Attributes:
        mask_ratio: Ratio of elements to mask (0.0-1.0)
        mask_types: Types of code elements to mask
    """

    mask_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Ratio of elements to mask",
    )
    mask_types: list[MaskType] = Field(
        default_factory=lambda: [
            MaskType.CONSTANTS,
            MaskType.CONDITIONS,
            MaskType.RETURNS,
        ],
        description="Types of code elements to mask",
    )


class ScenarioType(str, Enum):
    """Types of scenarios for walkthrough testing."""

    HAPPY_PATH = "happy_path"
    ERROR_PATH = "error_path"
    EDGE_CASE = "edge_case"


class ScenarioWalkthroughConfig(BaseModel):
    """Configuration for scenario walkthrough verification strategy.

    This strategy creates usage scenarios and validates that documentation
    correctly describes the behavior for each scenario.

    Attributes:
        scenarios_per_component: Number of scenarios per component
        types: Types of scenarios to generate
    """

    scenarios_per_component: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of scenarios per component",
    )
    types: list[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.HAPPY_PATH,
            ScenarioType.ERROR_PATH,
            ScenarioType.EDGE_CASE,
        ],
        description="Types of scenarios to generate",
    )


class MutationType(str, Enum):
    """Types of mutations for mutation detection testing."""

    BOUNDARY = "boundary"
    OFF_BY_ONE = "off_by_one"
    NULL_HANDLING = "null_handling"


class MutationDetectionConfig(BaseModel):
    """Configuration for mutation detection verification strategy.

    This strategy introduces mutations to code behavior descriptions
    and validates that the documentation can detect these changes.

    Attributes:
        mutations_per_component: Number of mutations per component
        mutation_types: Types of mutations to introduce
    """

    mutations_per_component: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of mutations per component",
    )
    mutation_types: list[MutationType] = Field(
        default_factory=lambda: [
            MutationType.BOUNDARY,
            MutationType.OFF_BY_ONE,
            MutationType.NULL_HANDLING,
        ],
        description="Types of mutations to introduce",
    )


class AdversarialReviewConfig(BaseModel):
    """Configuration for adversarial review verification strategy.

    This strategy uses an adversarial model to find issues
    in the documentation.

    Attributes:
        max_findings_per_component: Maximum findings to report per component
    """

    max_findings_per_component: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum findings per component",
    )


class TestGenerationConfig(BaseModel):
    """Configuration for test generation verification strategy.

    This strategy generates tests from documentation and validates
    they pass against the actual code.

    Attributes:
        tests_per_component: Number of tests to generate per component
        run_generated_tests: Whether to actually run the generated tests
    """

    tests_per_component: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of tests per component",
    )
    run_generated_tests: bool = Field(
        default=True,
        description="Whether to run generated tests",
    )


class VerificationConfig(BaseModel):
    """Configuration for the CrossCheck verification framework.

    The CrossCheck Verification Framework provides multiple strategies
    to validate documentation quality and accuracy against the source code.

    Attributes:
        enabled: Whether verification is enabled
        enabled_strategies: List of verification strategies to use
        thresholds: Score thresholds for passing verification
        qa_interrogation: Q&A interrogation strategy config
        masked_reconstruction: Masked reconstruction strategy config
        scenario_walkthrough: Scenario walkthrough strategy config
        mutation_detection: Mutation detection strategy config
        adversarial_review: Adversarial review strategy config
        test_generation: Test generation strategy config
    """

    enabled: bool = Field(
        default=False,
        description="Enable verification framework",
    )
    enabled_strategies: list[VerificationStrategy] = Field(
        default_factory=lambda: [
            VerificationStrategy.QA_INTERROGATION,
            VerificationStrategy.MASKED_RECONSTRUCTION,
            VerificationStrategy.SCENARIO_WALKTHROUGH,
            VerificationStrategy.MUTATION_DETECTION,
            VerificationStrategy.IMPACT_ANALYSIS,
            VerificationStrategy.ADVERSARIAL_REVIEW,
            VerificationStrategy.TEST_GENERATION,
        ],
        description="Verification strategies to run",
    )
    thresholds: VerificationThresholdsConfig = Field(
        default_factory=VerificationThresholdsConfig,
        description="Verification score thresholds",
    )
    qa_interrogation: QAInterrogationConfig = Field(
        default_factory=QAInterrogationConfig,
        description="Q&A interrogation config",
    )
    masked_reconstruction: MaskedReconstructionConfig = Field(
        default_factory=MaskedReconstructionConfig,
        description="Masked reconstruction config",
    )
    scenario_walkthrough: ScenarioWalkthroughConfig = Field(
        default_factory=ScenarioWalkthroughConfig,
        description="Scenario walkthrough config",
    )
    mutation_detection: MutationDetectionConfig = Field(
        default_factory=MutationDetectionConfig,
        description="Mutation detection config",
    )
    adversarial_review: AdversarialReviewConfig = Field(
        default_factory=AdversarialReviewConfig,
        description="Adversarial review config",
    )
    test_generation: TestGenerationConfig = Field(
        default_factory=TestGenerationConfig,
        description="Test generation config",
    )

    def is_strategy_enabled(self, strategy: VerificationStrategy) -> bool:
        """Check if a specific verification strategy is enabled.

        Args:
            strategy: The strategy to check

        Returns:
            True if the strategy is in the enabled_strategies list
        """
        return self.enabled and strategy in self.enabled_strategies


class LogLevel(str, Enum):
    """Logging level."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        file: Optional log file path (None for stdout only)
        json_output: Output logs as JSON for structured logging
        include_timestamp: Include timestamp in log output
        include_module: Include module name in log output
    """

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file: str | None = Field(
        default=None,
        description="Log file path (None for stdout only)",
    )
    json_output: bool = Field(
        default=False,
        description="Output logs as JSON",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in logs",
    )
    include_module: bool = Field(
        default=True,
        description="Include module name in logs",
    )

    def get_format_string(self) -> str:
        """Build format string based on configuration.

        Returns:
            Log format string
        """
        if self.json_output:
            # For JSON output, return a simple format (actual JSON formatting
            # would be handled by the logging handler)
            return "%(message)s"

        parts = []
        if self.include_timestamp:
            parts.append("%(asctime)s")
        if self.include_module:
            parts.append("%(name)s")
        parts.append("%(levelname)s")
        parts.append("%(message)s")

        return " - ".join(parts)


class OutputConfig(BaseModel):
    """Configuration for output paths.

    Attributes:
        base_dir: Base output directory
        documentation_file: Documentation output filename
        call_graph_file: Call graph output filename
        rebuild_tickets_file: Rebuild tickets filename
        convergence_report_file: Convergence report filename
        metrics_file: Metrics output filename
        create_dirs: Create directories if they don't exist
    """

    base_dir: str = Field(
        default="./output",
        description="Base output directory",
    )
    documentation_file: str = Field(
        default="documentation.json",
        description="Documentation filename",
    )
    call_graph_file: str = Field(
        default="call_graph.json",
        description="Call graph filename",
    )
    rebuild_tickets_file: str = Field(
        default="rebuild_tickets.json",
        description="Rebuild tickets filename",
    )
    convergence_report_file: str = Field(
        default="convergence_report.json",
        description="Convergence report filename",
    )
    metrics_file: str = Field(
        default="metrics.json",
        description="Metrics filename",
    )
    create_dirs: bool = Field(
        default=True,
        description="Create output directories",
    )

    def get_path(self, filename_attr: str) -> Path:
        """Get full path for an output file.

        Args:
            filename_attr: Attribute name for the filename

        Returns:
            Full Path object
        """
        filename = getattr(self, filename_attr)
        return Path(self.base_dir) / filename


class TwinscribeConfig(BaseModel):
    """Root configuration for the entire system.

    This is the main configuration class that aggregates all
    configuration sections.

    Attributes:
        codebase: Codebase configuration
        models: Model configuration
        convergence: Convergence criteria
        beads: Beads integration configuration
        static_analysis: Static analysis configuration
        verification: CrossCheck verification framework configuration
        output: Output paths configuration
        debug: Enable debug mode
        dry_run: Run without making changes
    """

    codebase: CodebaseConfig = Field(
        ...,
        description="Codebase configuration",
    )
    models: ModelsConfig = Field(
        default_factory=ModelsConfig,
        description="Model configuration",
    )
    convergence: ConvergenceConfig = Field(
        default_factory=ConvergenceConfig,
        description="Convergence criteria",
    )
    beads: BeadsConfig = Field(
        default_factory=BeadsConfig,
        description="Beads integration",
    )
    static_analysis: StaticAnalysisConfig = Field(
        default_factory=StaticAnalysisConfig,
        description="Static analysis config",
    )
    verification: VerificationConfig = Field(
        default_factory=VerificationConfig,
        description="CrossCheck verification framework config",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    dry_run: bool = Field(
        default=False,
        description="Run without making changes",
    )

    @model_validator(mode="after")
    def validate_model_references(self) -> "TwinscribeConfig":
        """Validate that all referenced models exist."""
        models = self.models

        # Check stream A models
        models.get_model_config(models.stream_a.documenter)
        models.get_model_config(models.stream_a.validator)

        # Check stream B models
        models.get_model_config(models.stream_b.documenter)
        models.get_model_config(models.stream_b.validator)

        # Check comparator model
        models.get_model_config(models.comparator)

        return self

    def to_yaml_dict(self) -> dict[str, Any]:
        """Convert to YAML-friendly dictionary.

        Returns:
            Dict suitable for YAML serialization
        """
        return self.model_dump(mode="json", exclude_none=True)
