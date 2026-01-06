"""
Configuration Data Models.

Defines all configuration schemas using Pydantic for validation
and type safety.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
DEFAULT_MODELS = {
    "claude-sonnet-4-5": ModelConfig(
        name="claude-sonnet-4-5-20250929",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.GENERATION,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        max_tokens=4096,
        context_window=200000,
    ),
    "claude-haiku-4-5": ModelConfig(
        name="claude-haiku-4-5-20251001",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.VALIDATION,
        cost_per_million_input=0.25,
        cost_per_million_output=1.25,
        max_tokens=2048,
        context_window=200000,
    ),
    "claude-opus-4-5": ModelConfig(
        name="claude-opus-4-5-20251101",
        provider=ModelProvider.OPENROUTER,
        tier=ModelTier.ARBITRATION,
        cost_per_million_input=15.0,
        cost_per_million_output=75.0,
        max_tokens=8192,
        context_window=200000,
    ),
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
            documenter="claude-sonnet-4-5",
            validator="claude-haiku-4-5",
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
        default="claude-opus-4-5",
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
    executable: Optional[str] = Field(default=None)
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
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration",
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
