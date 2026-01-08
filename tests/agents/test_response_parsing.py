"""
Tests for LLM response parsing edge cases.

These tests verify that the response parsing logic handles:
- Long summaries (truncation)
- Malformed JSON (repair)
- Scores on 0-10 scale (normalization)
- Missing/invalid fields (graceful handling)
"""

import json

import pytest

from twinscribe.agents.documenter import DocumenterConfig
from twinscribe.agents.validator import ValidatorConfig
from twinscribe.agents.stream import ConcreteDocumenterAgent, ConcreteValidatorAgent
from twinscribe.models.base import ModelTier, StreamId


@pytest.fixture
def documenter_config() -> DocumenterConfig:
    """Create a DocumenterConfig for testing."""
    return DocumenterConfig(
        agent_id="test-doc",
        stream_id=StreamId.STREAM_A,
        model_tier=ModelTier.GENERATION,
        provider="anthropic",
        model_name="claude-sonnet-4-5-20250929",
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
    )


@pytest.fixture
def validator_config() -> ValidatorConfig:
    """Create a ValidatorConfig for testing."""
    return ValidatorConfig(
        agent_id="test-val",
        stream_id=StreamId.STREAM_A,
        model_tier=ModelTier.VALIDATION,
        provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        cost_per_million_input=0.25,
        cost_per_million_output=1.25,
    )


class TestSummaryTruncation:
    """Tests for summary length truncation."""

    def test_long_summary_is_truncated(self, documenter_config):
        """Summary > 200 chars should be truncated with ellipsis."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # Create response with very long summary
        long_summary = "A" * 300
        response = json.dumps({
            "documentation": {
                "summary": long_summary,
                "description": "Test description",
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert len(result.documentation.summary) <= 200
        assert result.documentation.summary.endswith("...")

    def test_summary_at_limit_not_truncated(self, documenter_config):
        """Summary exactly 200 chars should not be truncated."""
        agent = ConcreteDocumenterAgent(documenter_config)

        summary = "A" * 200
        response = json.dumps({
            "documentation": {
                "summary": summary,
                "description": "Test",
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.documentation.summary == summary
        assert not result.documentation.summary.endswith("...")

    def test_truncation_at_word_boundary(self, documenter_config):
        """Long summaries should truncate at word boundary when possible."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # Create a summary with words that makes truncation point clear
        words = "word " * 50  # 250 chars
        response = json.dumps({
            "documentation": {
                "summary": words,
                "description": "Test",
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should end with "..." and not cut mid-word
        assert result.documentation.summary.endswith("...")
        # The part before "..." should end with a complete word
        before_ellipsis = result.documentation.summary[:-3]
        assert before_ellipsis.endswith("word") or before_ellipsis.endswith(" ")


class TestJsonRepair:
    """Tests for JSON syntax error repair."""

    def test_trailing_comma_in_object_is_fixed(self, documenter_config):
        """Trailing comma before } should be removed."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # Invalid JSON with trailing comma
        bad_json = '{"documentation": {"summary": "Test", "description": "Desc",}, "call_graph": {"callers": [], "callees": []}}'

        result = agent._parse_response(bad_json, "test.Component", 100)

        # Should parse successfully
        assert result.documentation.summary == "Test"

    def test_trailing_comma_in_array_is_fixed(self, documenter_config):
        """Trailing comma before ] should be removed."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # Invalid JSON with trailing comma in array
        bad_json = '{"documentation": {"summary": "Test", "description": "Desc", "examples": ["ex1", "ex2",]}, "call_graph": {"callers": [], "callees": []}}'

        result = agent._parse_response(bad_json, "test.Component", 100)

        assert result.documentation.summary == "Test"
        assert result.documentation.examples == ["ex1", "ex2"]

    def test_missing_comma_between_elements_is_fixed(self, documenter_config):
        """Missing comma between JSON elements should be added."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # Invalid JSON with missing comma (space between strings)
        bad_json = '{"documentation": {"summary": "Test"  "description": "Desc"}, "call_graph": {"callers": [], "callees": []}}'

        result = agent._parse_response(bad_json, "test.Component", 100)

        # Should parse with repair
        assert result.documentation.summary == "Test"


class TestScoreNormalization:
    """Tests for score normalization from 0-10 to 0-1 scale."""

    def test_score_on_0_10_scale_is_normalized(self, validator_config):
        """Score like 8 or 9 should be normalized to 0.8 or 0.9."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "passed",
            "completeness": {"score": 9, "missing_elements": []},
            "call_graph_accuracy": {"score": 8, "verified_callees": [], "missing_callees": []},
            "description_quality": {"score": 10, "issues": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # All scores should be normalized to 0-1 range
        assert 0 <= result.completeness.score <= 1
        assert result.completeness.score == 0.9
        assert result.call_graph_accuracy.score == 0.8
        assert result.description_quality.score == 1.0

    def test_score_on_0_1_scale_unchanged(self, validator_config):
        """Score already in 0-1 range should not be changed."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "passed",
            "completeness": {"score": 0.95, "missing_elements": []},
            "call_graph_accuracy": {"score": 0.85, "verified_callees": []},
            "description_quality": {"score": 1.0, "issues": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.completeness.score == 0.95
        assert result.call_graph_accuracy.score == 0.85
        assert result.description_quality.score == 1.0

    def test_confidence_score_normalized(self, documenter_config):
        """Confidence score on 0-10 scale should be normalized."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {"callers": [], "callees": []},
            "confidence": 9,  # 0-10 scale
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.metadata.confidence == 0.9


class TestTypeGuards:
    """Tests for type guard handling of unexpected types."""

    def test_parameters_as_strings_handled(self, documenter_config):
        """Parameters returned as strings instead of dicts should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "parameters": ["param1", "param2"],  # Strings instead of dicts
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should handle gracefully - use string as name
        assert len(result.documentation.parameters) == 2
        assert result.documentation.parameters[0].name == "param1"

    def test_call_graph_as_string_handled(self, documenter_config):
        """Call graph returned as string instead of dict should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": "no dependencies",  # String instead of dict
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should have empty call graph, not crash
        assert len(result.call_graph.callers) == 0
        assert len(result.call_graph.callees) == 0

    def test_nested_data_as_string_handled(self, validator_config):
        """Nested data returned as string should be handled."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "passed",
            "completeness": "all complete",  # String instead of dict
            "call_graph_accuracy": "accurate",  # String instead of dict
            "description_quality": "good",  # String instead of dict
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should use defaults, not crash
        assert result.completeness.score == 1.0  # Default
        assert result.call_graph_accuracy.score == 1.0  # Default
        assert result.description_quality.score == 1.0  # Default

    def test_description_as_dict_handled(self, documenter_config):
        """Description returned as dict instead of string should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": {
                    "what": "This class implements logging",
                    "how": "By printing to stdout",
                },
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should convert dict to string by joining values
        assert isinstance(result.documentation.description, str)
        assert "logging" in result.documentation.description
        assert "stdout" in result.documentation.description

    def test_summary_as_dict_handled(self, documenter_config):
        """Summary returned as dict instead of string should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": {"brief": "A utility class"},
                "description": "Detailed description here",
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should convert dict to string
        assert isinstance(result.documentation.summary, str)
        assert "utility" in result.documentation.summary
