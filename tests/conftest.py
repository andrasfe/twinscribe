"""
TwinScribe Test Configuration and Fixtures

This module provides pytest fixtures for testing the dual-stream documentation system.
All fixtures are designed to avoid real API calls and provide deterministic behavior.

Fixture Categories:
- Mock LLM Clients: Simulated responses for Anthropic and OpenAI APIs
- Sample Code: Python code snippets for testing AST parsing and analysis
- Call Graphs: Pre-computed call graph data for testing static analysis
- Documentation Models: Sample documentation outputs for testing validation
"""

import asyncio
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_codebase_dir(fixtures_dir: Path) -> Path:
    """Return the sample codebase directory for testing."""
    return fixtures_dir / "sample_codebase"


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock LLM Client Fixtures
# =============================================================================


class MockLLMResponse:
    """Mock response object for LLM API calls."""

    def __init__(self, content: str, model: str = "mock-model"):
        self.content = content
        self.model = model
        self.id = "mock-response-id"
        self.created = int(datetime.now(UTC).timestamp())
        self.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    @property
    def choices(self) -> list:
        """OpenAI-compatible choices format."""
        return [MagicMock(message=MagicMock(content=self.content))]


class MockAnthropicResponse:
    """Mock response for Anthropic API calls."""

    def __init__(self, content: str, model: str = "claude-sonnet-4-5-20250929"):
        self.id = "msg_mock_" + "a" * 20
        self.type = "message"
        self.role = "assistant"
        self.model = model
        self.content = [MagicMock(type="text", text=content)]
        self.stop_reason = "end_turn"
        self.usage = MagicMock(input_tokens=100, output_tokens=50)


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """
    Create a mock OpenAI client that returns predictable responses.

    Usage:
        def test_something(mock_openai_client):
            # Configure specific response
            mock_openai_client.chat.completions.create.return_value = ...
    """
    client = MagicMock()

    # Default response
    default_response = MockLLMResponse(content='{"status": "success", "message": "Mock response"}')
    client.chat.completions.create = AsyncMock(return_value=default_response)

    return client


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """
    Create a mock Anthropic client that returns predictable responses.

    Usage:
        def test_something(mock_anthropic_client):
            # Configure specific response
            mock_anthropic_client.messages.create.return_value = ...
    """
    client = MagicMock()

    # Default response
    default_response = MockAnthropicResponse(
        content='{"status": "success", "message": "Mock response"}'
    )
    client.messages.create = AsyncMock(return_value=default_response)

    return client


@pytest.fixture
def mock_documenter_response() -> dict:
    """
    Return a mock documentation output matching the DocumentationOutput schema.
    """
    return {
        "component_id": "sample_module.SampleClass.sample_method",
        "documentation": {
            "summary": "Process input data and return results.",
            "description": "A comprehensive method that processes the given input data, "
            "performs validation, and returns the transformed results.",
            "parameters": [
                {
                    "name": "data",
                    "type": "dict[str, Any]",
                    "description": "Input data to process",
                    "default": None,
                },
                {
                    "name": "options",
                    "type": "ProcessingOptions | None",
                    "description": "Optional processing configuration",
                    "default": "None",
                },
            ],
            "returns": {
                "type": "ProcessingResult",
                "description": "The processed result containing transformed data",
            },
            "raises": [
                {"type": "ValueError", "condition": "When data is empty or malformed"},
                {
                    "type": "ProcessingError",
                    "condition": "When processing fails due to invalid options",
                },
            ],
            "examples": [
                'result = sample.sample_method({"key": "value"})',
                "result = sample.sample_method(data, options=ProcessingOptions(strict=True))",
            ],
        },
        "call_graph": {
            "callers": [
                {
                    "component_id": "sample_module.Client.execute",
                    "call_site_line": 45,
                    "call_type": "direct",
                }
            ],
            "callees": [
                {
                    "component_id": "sample_module.Validator.validate",
                    "call_site_line": 12,
                    "call_type": "direct",
                },
                {
                    "component_id": "sample_module.Transformer.transform",
                    "call_site_line": 15,
                    "call_type": "conditional",
                },
            ],
        },
        "metadata": {
            "agent_id": "A1",
            "model": "claude-sonnet-4-5-20250929",
            "timestamp": "2026-01-06T10:00:00Z",
            "confidence": 0.92,
            "processing_order": 1,
        },
    }


@pytest.fixture
def mock_validation_response() -> dict:
    """
    Return a mock validation output matching the ValidationResult schema.
    """
    return {
        "component_id": "sample_module.SampleClass.sample_method",
        "validation_result": "pass",
        "completeness": {"score": 0.95, "missing_elements": [], "extra_elements": []},
        "call_graph_accuracy": {
            "score": 0.98,
            "verified_callees": [
                "sample_module.Validator.validate",
                "sample_module.Transformer.transform",
            ],
            "missing_callees": [],
            "false_callees": [],
            "verified_callers": ["sample_module.Client.execute"],
            "missing_callers": [],
            "false_callers": [],
        },
        "corrections_applied": [],
        "metadata": {
            "agent_id": "A2",
            "model": "claude-haiku-4-5-20251001",
            "static_analyzer": "pycg",
            "timestamp": "2026-01-06T10:01:00Z",
        },
    }


@pytest.fixture
def mock_comparison_response() -> dict:
    """
    Return a mock comparison output matching the ComparisonResult schema.
    """
    return {
        "comparison_id": "cmp_20260106_001",
        "iteration": 1,
        "summary": {
            "total_components": 10,
            "identical": 8,
            "discrepancies": 2,
            "resolved_by_ground_truth": 1,
            "requires_human_review": 1,
        },
        "discrepancies": [
            {
                "discrepancy_id": "disc_001",
                "component_id": "sample_module.Helper.process",
                "type": "call_graph_edge",
                "stream_a_value": {"callee": "utils.helper", "line": 20},
                "stream_b_value": None,
                "ground_truth": {"callee": "utils.helper", "line": 20},
                "resolution": "accept_stream_a",
                "confidence": 0.99,
                "requires_beads": False,
            }
        ],
        "convergence_status": {
            "converged": False,
            "blocking_discrepancies": 1,
            "recommendation": "continue_iteration",
        },
        "metadata": {
            "agent_id": "C",
            "model": "claude-opus-4-5-20251101",
            "timestamp": "2026-01-06T10:02:00Z",
            "comparison_duration_ms": 2500,
        },
    }


# =============================================================================
# Sample Code Fixtures
# =============================================================================


@pytest.fixture
def sample_python_function() -> str:
    """Return a sample Python function for testing."""
    return '''
def calculate_total(items: list[dict], tax_rate: float = 0.1) -> float:
    """
    Calculate the total price of items including tax.

    Args:
        items: List of items with 'price' and 'quantity' keys
        tax_rate: Tax rate to apply (default 10%)

    Returns:
        Total price including tax

    Raises:
        ValueError: If items list is empty
        KeyError: If item is missing required keys
    """
    if not items:
        raise ValueError("Items list cannot be empty")

    subtotal = 0.0
    for item in items:
        subtotal += item["price"] * item["quantity"]

    return subtotal * (1 + tax_rate)
'''


@pytest.fixture
def sample_python_class() -> str:
    """Return a sample Python class for testing."""
    return '''
class DataProcessor:
    """Process and transform data with validation."""

    def __init__(self, config: dict | None = None):
        """Initialize processor with optional config."""
        self.config = config or {}
        self._cache = {}

    def process(self, data: dict) -> dict:
        """
        Process input data and return transformed result.

        Args:
            data: Input data dictionary

        Returns:
            Transformed data dictionary

        Raises:
            ValueError: If data validation fails
        """
        self._validate(data)
        result = self._transform(data)
        self._cache[id(data)] = result
        return result

    def _validate(self, data: dict) -> None:
        """Validate input data."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

    def _transform(self, data: dict) -> dict:
        """Apply transformations to data."""
        return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}
'''


@pytest.fixture
def sample_python_module() -> str:
    """Return a sample Python module with multiple components for testing."""
    return '''
"""Sample module for testing documentation generation."""

from typing import Any


def helper_function(value: Any) -> str:
    """Convert value to string representation."""
    return str(value)


class Calculator:
    """Simple calculator with basic operations."""

    def __init__(self, precision: int = 2):
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        return round(result, self.precision)

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        formatted = helper_function(a * b)
        return round(float(formatted), self.precision)


class AdvancedCalculator(Calculator):
    """Extended calculator with advanced operations."""

    def power(self, base: float, exponent: float) -> float:
        """Calculate base raised to exponent."""
        result = base ** exponent
        return round(result, self.precision)

    def compute_complex(self, values: list[float]) -> float:
        """Perform complex computation on list of values."""
        total = 0.0
        for v in values:
            total = self.add(total, v)
        return self.multiply(total, len(values))
'''


# =============================================================================
# Call Graph Fixtures
# =============================================================================


@pytest.fixture
def sample_call_graph() -> dict:
    """Return a sample call graph for testing."""
    return {
        "edges": [
            {
                "caller": "sample_module.Calculator.multiply",
                "callee": "sample_module.helper_function",
                "call_site_line": 22,
                "call_type": "direct",
            },
            {
                "caller": "sample_module.AdvancedCalculator.compute_complex",
                "callee": "sample_module.Calculator.add",
                "call_site_line": 35,
                "call_type": "loop",
            },
            {
                "caller": "sample_module.AdvancedCalculator.compute_complex",
                "callee": "sample_module.Calculator.multiply",
                "call_site_line": 36,
                "call_type": "direct",
            },
        ],
        "nodes": [
            "sample_module.helper_function",
            "sample_module.Calculator.__init__",
            "sample_module.Calculator.add",
            "sample_module.Calculator.multiply",
            "sample_module.AdvancedCalculator.power",
            "sample_module.AdvancedCalculator.compute_complex",
        ],
    }


@pytest.fixture
def empty_call_graph() -> dict:
    """Return an empty call graph for testing edge cases."""
    return {"edges": [], "nodes": []}


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def mock_config() -> dict:
    """Return a mock configuration dictionary."""
    return {
        "codebase": {
            "path": "/tmp/test_codebase",
            "language": "python",
            "exclude_patterns": ["**/test_*", "**/tests/**"],
        },
        "models": {
            "stream_a": {
                "documenter": "claude-sonnet-4-5-20250929",
                "validator": "claude-haiku-4-5-20251001",
            },
            "stream_b": {"documenter": "gpt-4o", "validator": "gpt-4o-mini"},
            "comparator": "claude-opus-4-5-20251101",
        },
        "convergence": {
            "max_iterations": 5,
            "call_graph_match_threshold": 0.98,
            "documentation_similarity_threshold": 0.95,
        },
        "beads": {
            "server": "https://test.atlassian.net",
            "project": "TEST_DOC",
            "rebuild_project": "TEST_REBUILD",
        },
        "static_analysis": {"python": {"tool": "pycg", "fallback": "pyan3"}},
        "output": {
            "documentation_path": "./output/documentation.json",
            "call_graph_path": "./output/call_graph.json",
        },
    }


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Set up mock environment variables for testing."""
    env_vars = {
        "OPENROUTER_API_KEY": "test-openrouter-key-12345",
        "ANTHROPIC_API_KEY": "test-anthropic-key-12345",
        "OPENAI_API_KEY": "test-openai-key-12345",
        "BEADS_API_TOKEN": "test-beads-token-12345",
        "BEADS_SERVER": "https://test.atlassian.net",
        "BEADS_USERNAME": "test@example.com",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# =============================================================================
# Beads/Issue Tracking Fixtures
# =============================================================================


@pytest.fixture
def mock_beads_client() -> MagicMock:
    """Create a mock Beads client for testing issue operations."""
    client = MagicMock()

    # Mock issue creation
    mock_issue = MagicMock()
    mock_issue.id = "bd-a1b2"
    mock_issue.title = "Test Issue"
    mock_issue.status = "open"
    client.create_issue.return_value = mock_issue

    # Mock issue retrieval
    client.get_issue.return_value = mock_issue

    # Mock list ready
    client.list_ready.return_value = [mock_issue]

    return client


@pytest.fixture
def sample_discrepancy_ticket() -> dict:
    """Return a sample discrepancy ticket for testing."""
    return {
        "project": "TEST_DOC",
        "issue_type": "Clarification",
        "priority": "Medium",
        "summary": "[AI-DOC] documentation_content: sample_module.process",
        "description": """## Discrepancy Summary

**Component:** `sample_module.process`
**File:** `sample_module.py:10-25`
**Iteration:** 1

## Stream Comparison

| Aspect | Stream A | Stream B |
|--------|----------|----------|
| Summary | Processes data synchronously | Processes data with async option |

## Requested Action

Please review and indicate which interpretation is correct.
""",
        "labels": ["ai-documentation", "documentation_content", "iteration-1"],
    }


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def async_mock() -> AsyncMock:
    """Return a fresh AsyncMock for testing async code."""
    return AsyncMock()


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================


@pytest.fixture
def temp_codebase(tmp_path: Path, sample_python_module: str) -> Path:
    """Create a temporary codebase directory with sample files."""
    codebase_dir = tmp_path / "sample_codebase"
    codebase_dir.mkdir()

    # Create sample module file
    (codebase_dir / "sample_module.py").write_text(sample_python_module)

    # Create __init__.py
    (codebase_dir / "__init__.py").write_text('"""Sample package."""\n')

    return codebase_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for test results."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Model Instance Fixtures
# =============================================================================


@pytest.fixture
def sample_component_location() -> dict:
    """Return a sample component location dict."""
    return {
        "file_path": "sample_module.py",
        "line_start": 10,
        "line_end": 25,
        "column_start": 0,
        "column_end": None,
    }


@pytest.fixture
def sample_parameter_doc() -> dict:
    """Return a sample parameter documentation dict."""
    return {
        "name": "data",
        "type": "dict[str, Any]",
        "description": "Input data to process",
        "default": None,
    }


@pytest.fixture
def sample_component() -> dict:
    """Return a sample component dict for testing."""
    return {
        "component_id": "sample_module.SampleClass.process",
        "component_type": "method",
        "name": "process",
        "qualified_name": "sample_module.SampleClass.process",
        "location": {
            "file_path": "sample_module.py",
            "line_start": 15,
            "line_end": 30,
            "column_start": 4,
            "column_end": None,
        },
        "source_code": "def process(self, data: dict) -> dict:\n    ...",
        "parent_class": "SampleClass",
        "decorators": [],
        "is_async": False,
        "is_generator": False,
        "docstring": "Process input data.",
    }


# =============================================================================
# Fixture File Loading
# =============================================================================


@pytest.fixture
def llm_responses_dir(fixtures_dir: Path) -> Path:
    """Return the LLM responses fixtures directory."""
    return fixtures_dir / "llm_responses"


@pytest.fixture
def static_analysis_dir(fixtures_dir: Path) -> Path:
    """Return the static analysis fixtures directory."""
    return fixtures_dir / "static_analysis"


@pytest.fixture
def discrepancies_dir(fixtures_dir: Path) -> Path:
    """Return the discrepancies fixtures directory."""
    return fixtures_dir / "discrepancies"


@pytest.fixture
def pycg_output(static_analysis_dir: Path) -> dict:
    """Load the PyCG output fixture."""
    with open(static_analysis_dir / "pycg_output.json") as f:
        return json.load(f)


@pytest.fixture
def call_graph_ground_truth(static_analysis_dir: Path) -> dict:
    """Load the ground truth call graph fixture."""
    with open(static_analysis_dir / "call_graph_ground_truth.json") as f:
        return json.load(f)


@pytest.fixture
def documenter_responses(llm_responses_dir: Path) -> dict:
    """Load all documenter response fixtures."""
    with open(llm_responses_dir / "documenter_responses.json") as f:
        return json.load(f)


@pytest.fixture
def validator_responses(llm_responses_dir: Path) -> dict:
    """Load all validator response fixtures."""
    with open(llm_responses_dir / "validator_responses.json") as f:
        return json.load(f)


@pytest.fixture
def comparator_responses(llm_responses_dir: Path) -> dict:
    """Load all comparator response fixtures."""
    with open(llm_responses_dir / "comparator_responses.json") as f:
        return json.load(f)


@pytest.fixture
def call_graph_discrepancy_scenarios(discrepancies_dir: Path) -> list:
    """Load call graph discrepancy scenarios."""
    with open(discrepancies_dir / "call_graph_discrepancies.json") as f:
        data = json.load(f)
        return data.get("scenarios", [])


@pytest.fixture
def documentation_discrepancy_scenarios(discrepancies_dir: Path) -> list:
    """Load documentation discrepancy scenarios."""
    with open(discrepancies_dir / "documentation_discrepancies.json") as f:
        data = json.load(f)
        return data.get("scenarios", [])


@pytest.fixture
def mixed_discrepancy_scenarios(discrepancies_dir: Path) -> list:
    """Load mixed discrepancy scenarios."""
    with open(discrepancies_dir / "mixed_discrepancies.json") as f:
        data = json.load(f)
        return data.get("scenarios", [])
