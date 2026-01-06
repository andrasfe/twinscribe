# Configuration Management Design Document

## Overview

This document describes the configuration management system for the Dual-Stream Documentation System. The system supports YAML configuration files with environment variable substitution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Configuration System                              │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                      ConfigLoader                                  │ │
│  │                                                                    │ │
│  │   config.yaml ─────▶ YAML Parser ─────▶ Env Substitution ────┐   │ │
│  │                                                               │   │ │
│  │   .env ────────────▶ EnvironmentLoader ──────────────────────┤   │ │
│  │                                                               │   │ │
│  │                                                               ▼   │ │
│  │                                              ┌──────────────────┐ │ │
│  │                                              │  Pydantic        │ │ │
│  │                                              │  Validation      │ │ │
│  │                                              └────────┬─────────┘ │ │
│  │                                                       │           │ │
│  └───────────────────────────────────────────────────────┼───────────┘ │
│                                                          │             │
│                                                          ▼             │
│                                              ┌──────────────────────┐  │
│                                              │   TwinscribeConfig   │  │
│                                              │                      │  │
│                                              │ - codebase           │  │
│                                              │ - models             │  │
│                                              │ - convergence        │  │
│                                              │ - beads              │  │
│                                              │ - static_analysis    │  │
│                                              │ - output             │  │
│                                              └──────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/twinscribe/config/
├── __init__.py        # Public exports
├── models.py          # Configuration data models
├── environment.py     # Environment variable handling
└── loader.py          # YAML loading and validation
```

## Configuration Files

### .env File

Environment variables for secrets and local overrides.

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional fallbacks
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key

# Beads integration (git-backed issue tracker)
# Run 'bd init' in the project root to initialize
BEADS_DIR=.beads
BEADS_AUTO_CREATE=true
```

### config.yaml File

Main configuration in YAML format.

```yaml
codebase:
  path: /path/to/codebase
  language: python
  exclude_patterns:
    - "**/test_*"
    - "**/tests/**"

models:
  stream_a:
    documenter: claude-sonnet-4-5
    validator: claude-haiku-4-5
  stream_b:
    documenter: gpt-4o
    validator: gpt-4o-mini
  comparator: claude-opus-4-5

convergence:
  max_iterations: 5
  call_graph_match_threshold: 0.98

beads:
  enabled: true
  directory: ${BEADS_DIR:.beads}

output:
  base_dir: ./output
```

## Configuration Models

### TwinscribeConfig (Root)

```python
class TwinscribeConfig(BaseModel):
    codebase: CodebaseConfig         # Required
    models: ModelsConfig             # Default provided
    convergence: ConvergenceConfig   # Default provided
    beads: BeadsConfig               # Default provided
    static_analysis: StaticAnalysisConfig
    output: OutputConfig
    debug: bool = False
    dry_run: bool = False
```

### CodebaseConfig

```python
class CodebaseConfig(BaseModel):
    path: str                    # Required
    language: Language = PYTHON
    exclude_patterns: list[str]  # Default patterns
    include_patterns: list[str] = []
    entry_points: list[str] = []
```

### ModelsConfig

```python
class ModelsConfig(BaseModel):
    stream_a: StreamModelsConfig
    stream_b: StreamModelsConfig
    comparator: str = "claude-opus-4-5"
    custom_models: dict[str, ModelConfig] = {}

class StreamModelsConfig(BaseModel):
    documenter: str  # Model name or alias
    validator: str   # Model name or alias
```

### ModelConfig

```python
class ModelConfig(BaseModel):
    name: str                    # Full model identifier
    provider: ModelProvider      # openrouter, anthropic, openai
    tier: ModelTier              # generation, validation, arbitration
    cost_per_million_input: float
    cost_per_million_output: float
    max_tokens: int = 4096
    temperature: float = 0.0
    context_window: int = 128000
```

### Default Models

| Alias | Model Name | Provider | Tier | Input Cost | Output Cost |
|-------|------------|----------|------|------------|-------------|
| claude-sonnet-4-5 | claude-sonnet-4-5-20250929 | openrouter | generation | $3.00/M | $15.00/M |
| claude-haiku-4-5 | claude-haiku-4-5-20251001 | openrouter | validation | $0.25/M | $1.25/M |
| claude-opus-4-5 | claude-opus-4-5-20251101 | openrouter | arbitration | $15.00/M | $75.00/M |
| gpt-4o | gpt-4o | openrouter | generation | $2.50/M | $10.00/M |
| gpt-4o-mini | gpt-4o-mini | openrouter | validation | $0.15/M | $0.60/M |

### ConvergenceConfig

```python
class ConvergenceConfig(BaseModel):
    max_iterations: int = 5
    call_graph_match_threshold: float = 0.98
    documentation_similarity_threshold: float = 0.95
    max_open_discrepancies: int = 2
    beads_ticket_timeout_hours: int = 48
```

### BeadsConfig

```python
class BeadsConfig(BaseModel):
    enabled: bool = True
    directory: str = ".beads"
    labels: list[str] = ["ai-documentation", "twinscribe"]
    auto_create_issues: bool = True
    discrepancy_priority: int = 1
    rebuild_priority: int = 0
```

### StaticAnalysisConfig

```python
class StaticAnalysisConfig(BaseModel):
    python: dict[str, AnalyzerToolConfig]
    java: dict[str, AnalyzerToolConfig]
    javascript: dict[str, AnalyzerToolConfig]
    multi_language_fallback: AnalyzerToolConfig
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

class AnalyzerToolConfig(BaseModel):
    enabled: bool = True
    executable: Optional[str] = None
    extra_args: list[str] = []
    timeout_seconds: int = 300
```

### OutputConfig

```python
class OutputConfig(BaseModel):
    base_dir: str = "./output"
    documentation_file: str = "documentation.json"
    call_graph_file: str = "call_graph.json"
    rebuild_tickets_file: str = "rebuild_tickets.json"
    convergence_report_file: str = "convergence_report.json"
    metrics_file: str = "metrics.json"
    create_dirs: bool = True
```

## Environment Variable Substitution

The configuration loader supports environment variable substitution:

```yaml
# Basic substitution
directory: ${BEADS_DIR}

# With default value
directory: ${BEADS_DIR:.beads}
```

Pattern: `${VAR_NAME}` or `${VAR_NAME:default}`

## Loader Interface

### ConfigLoader

```python
class ConfigLoader:
    def __init__(config_path: Optional[Path], env_file: str = ".env")

    def load() -> TwinscribeConfig
    def save(path: Optional[Path] = None) -> None

    @property config_path: Optional[Path]
    @property config: Optional[TwinscribeConfig]
```

### Convenience Functions

```python
# Load configuration
def load_config(config_path: Optional[Path], env_file: str = ".env") -> TwinscribeConfig

# Get loaded configuration
def get_config() -> TwinscribeConfig

# Reset (for testing)
def reset_config() -> None

# Create with defaults
def create_default_config(codebase_path: str) -> TwinscribeConfig
```

## Environment Interface

### EnvironmentConfig

```python
class EnvironmentConfig(BaseModel):
    openrouter_api_key: Optional[SecretStr]
    anthropic_api_key: Optional[SecretStr]
    openai_api_key: Optional[SecretStr]
    beads_api_token: Optional[SecretStr]
    beads_username: Optional[str]
    env_file: str = ".env"

    @property has_openrouter: bool
    @property has_anthropic: bool
    @property has_openai: bool
    @property has_beads: bool
```

### Environment Functions

```python
# Load environment
def load_environment(env_file: str = ".env") -> EnvironmentConfig

# Get API key
def get_api_key(provider: str) -> Optional[str]

# Get Beads credentials
def get_beads_credentials() -> tuple[Optional[str], Optional[str]]

# Validate environment
def validate_environment(require_openrouter: bool = True) -> list[str]
```

## Usage Examples

### Loading Configuration

```python
from twinscribe.config import load_config, get_config

# Load from file
config = load_config("config.yaml")

# Access configuration
print(f"Codebase: {config.codebase.path}")
print(f"Language: {config.codebase.language}")

# Get model config
model = config.models.get_model_config("claude-sonnet-4-5")
print(f"Model: {model.name}, Cost: ${model.cost_per_million_input}/M")
```

### Programmatic Configuration

```python
from twinscribe.config import (
    TwinscribeConfig,
    CodebaseConfig,
    ModelsConfig,
    StreamModelsConfig,
)

config = TwinscribeConfig(
    codebase=CodebaseConfig(
        path="/my/codebase",
        language="python",
    ),
    models=ModelsConfig(
        stream_a=StreamModelsConfig(
            documenter="claude-sonnet-4-5",
            validator="claude-haiku-4-5",
        ),
        stream_b=StreamModelsConfig(
            documenter="gpt-4o",
            validator="gpt-4o-mini",
        ),
    ),
)
```

### Environment Handling

```python
from twinscribe.config import load_environment, get_api_key

# Load .env file
env = load_environment(".env")

# Check what's configured
if env.has_openrouter:
    key = get_api_key("openrouter")
    # Use key...

# Validate required vars
missing = validate_environment(require_openrouter=True)
if missing:
    print(f"Missing environment variables: {missing}")
```

### Custom Models

```yaml
models:
  custom_models:
    my-fine-tuned:
      name: "ft:gpt-4o-mini:my-org::abc123"
      provider: openai
      tier: generation
      cost_per_million_input: 5.0
      cost_per_million_output: 15.0
      max_tokens: 4096

  stream_a:
    documenter: my-fine-tuned  # Use custom model
    validator: claude-haiku-4-5
```

## Validation

Configuration is validated at load time using Pydantic:

1. **Type validation**: All fields must match expected types
2. **Required fields**: `codebase.path` is required
3. **Range validation**: Scores must be 0.0-1.0, iterations >= 1
4. **Model references**: All referenced models must exist
5. **Path validation**: Codebase path must not be empty

### Validation Errors

```python
from twinscribe.config import load_config, ConfigurationError

try:
    config = load_config("config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    for error in e.errors:
        print(f"  - {error['loc']}: {error['msg']}")
```

## File Locations

| File | Purpose | Required |
|------|---------|----------|
| `.env` | Secrets and API keys | Yes (for API access) |
| `config.yaml` | Main configuration | No (defaults used) |
| `.env.example` | Template for .env | No (documentation) |
| `config.example.yaml` | Template for config | No (documentation) |

## Security

- API keys are stored in `.env`, never in YAML
- `SecretStr` type prevents accidental logging of secrets
- `.env` should be in `.gitignore`
- Environment variables take precedence over file values
