# TwinScribe

**Dual-Stream Code Documentation System with Tiered Model Architecture**

TwinScribe is a multi-agent system for generating accurate code documentation with call graph linkages. Two independent agent streams document and validate code in parallel, with discrepancies resolved through a premium-tier arbitrator agent that generates issue tickets for human review.

## Key Features

- **Dual Documentation Streams**: Two independent LLM streams generate documentation in parallel for increased accuracy
- **Tiered Model Architecture**: 90% cost reduction by using cheaper models for generation, premium models for judgment
- **Static Analysis Anchoring**: Call graphs are validated against AST-based static analysis (ground truth)
- **Beads Integration**: Human-in-the-loop resolution for edge cases with full audit trail

## Architecture Overview

```
                           SOURCE CODE
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
    +-----------+        +------------+        +-----------+
    | Stream A  |        | Ground     |        | Stream B  |
    | Claude    |        | Truth      |        | GPT-4     |
    | Sonnet    |        | (PyCG)     |        |           |
    +-----------+        +------------+        +-----------+
          |                     |                     |
          v                     v                     v
    +-----------+               |              +-----------+
    | Validator |               |              | Validator |
    | Claude    |               |              | GPT-4o    |
    | Haiku     |               |              | mini      |
    +-----------+               |              +-----------+
          |                     |                     |
          +----------+----------+----------+----------+
                     |                     |
                     v                     v
              +-------------+       +-------------+
              | Comparator  |       | Beads       |
              | Claude Opus |------>| Tickets     |
              +-------------+       +-------------+
                     |
                     v
              +-------------+
              | Final Docs  |
              +-------------+
```

## Requirements

- Python 3.11 or higher
- OpenRouter API key (for LLM access)
- Beads CLI (for issue tracking)

### Installing Beads

```bash
# Using the provided install script
./scripts/install-beads.sh

# Or manually via npm/Homebrew/Go:
npm install -g beads-cli
# or: brew install steveyegge/tap/beads
# or: go install github.com/steveyegge/beads/cmd/bd@latest
```

See https://github.com/steveyegge/beads for more details.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/twinscribe/twinscribe.git
cd twinscribe

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Using pip (when published)

```bash
pip install twinscribe
```

## Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Edit `.env` and fill in your credentials:

```bash
# Required: OpenRouter API key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Initialize Beads for issue tracking (run in project root)
# bd init
```

3. (Optional) Create a `config.yaml` for advanced configuration:

```yaml
codebase:
  path: /path/to/your/codebase
  language: python
  exclude_patterns:
    - "**/test_*"
    - "**/tests/**"
    - "**/__pycache__/**"

convergence:
  max_iterations: 5
  call_graph_match_threshold: 0.98
```

## Usage

### Command Line

```bash
# Document a Python codebase
twinscribe document /path/to/codebase --language python

# With custom configuration
twinscribe document /path/to/codebase --config config.yaml

# Generate only call graph (no documentation)
twinscribe analyze /path/to/codebase --output call_graph.json
```

### Python API

```python
import asyncio
from twinscribe import DualStreamOrchestrator

async def main():
    orchestrator = DualStreamOrchestrator(
        codebase_path="/path/to/codebase",
        language="python",
    )

    result = await orchestrator.run()

    print(f"Components documented: {len(result.documentation['components'])}")
    print(f"Call graph edges: {len(result.call_graph['edges'])}")
    print(f"Rebuild tickets: {len(result.rebuild_tickets)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Model Tiers

TwinScribe uses a tiered model architecture for cost optimization:

| Tier | Models | Cost | Use Case |
|------|--------|------|----------|
| **Generation** | Claude Sonnet 4.5, GPT-4o | ~$3/M tokens | Documentation writing |
| **Validation** | Claude Haiku 4.5, GPT-4o-mini | ~$0.20/M tokens | Verification checks |
| **Arbitration** | Claude Opus 4.5 | ~$15/M tokens | Comparison, judgment |

**Cost Projection (1000 components):** ~$4.70 vs ~$45.00 for all-premium baseline (90% savings)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=twinscribe --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Type Checking

```bash
mypy src/twinscribe
```

### Linting

```bash
ruff check src/twinscribe
ruff format src/twinscribe
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Project Structure

```
twinscribe/
├── src/
│   └── twinscribe/
│       ├── __init__.py
│       ├── version.py
│       ├── cli.py              # Command-line interface
│       ├── agents/             # Agent implementations
│       │   ├── __init__.py
│       │   ├── documenter.py   # Documentation generation
│       │   ├── validator.py    # Validation logic
│       │   └── comparator.py   # Comparison and arbitration
│       ├── static_analysis/    # Static analysis tools
│       │   ├── __init__.py
│       │   ├── oracle.py       # Ground truth provider
│       │   └── pycg.py         # PyCG integration
│       ├── beads/              # Issue tracking integration
│       │   ├── __init__.py
│       │   └── lifecycle.py    # Ticket management
│       ├── models/             # Pydantic data models
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── components.py
│       │   └── ...
│       └── utils/              # Shared utilities
│           ├── __init__.py
│           ├── config.py
│           └── logging.py
├── tests/                      # Test suite
├── pyproject.toml              # Project configuration
├── .env.example                # Environment template
├── .gitignore
└── README.md
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.
