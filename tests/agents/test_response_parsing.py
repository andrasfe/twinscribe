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

    def test_examples_as_string_handled(self, documenter_config):
        """Examples returned as string instead of list should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "examples": "print('hello world')",  # String instead of list
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should wrap in list
        assert isinstance(result.documentation.examples, list)
        assert len(result.documentation.examples) == 1
        assert "hello world" in result.documentation.examples[0]

    def test_invalid_call_type_defaults_to_direct(self, documenter_config):
        """Invalid call_type values should default to 'direct'."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {
                "callers": [
                    {"component_id": "foo.bar", "call_type": "inherited"},  # Invalid
                ],
                "callees": [
                    {"component_id": "baz.qux", "call_type": "indirect"},  # Invalid
                ],
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should default to DIRECT, not crash
        from twinscribe.models.base import CallType
        assert result.call_graph.callers[0].call_type == CallType.DIRECT
        assert result.call_graph.callees[0].call_type == CallType.DIRECT

    def test_call_site_line_as_string_handled(self, documenter_config):
        """call_site_line as string should be converted to int."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {
                "callers": [
                    {"component_id": "foo.bar", "call_site_line": "42"},  # String
                ],
                "callees": [
                    {"component_id": "baz.qux", "call_site_line": "100"},  # String
                ],
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should convert to int
        assert result.call_graph.callers[0].call_site_line == 42
        assert result.call_graph.callees[0].call_site_line == 100

    def test_documentation_as_string_handled(self, documenter_config):
        """documentation field as string instead of dict should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": "This is a simple utility function.",
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should use string as description
        assert "utility function" in result.documentation.description

    def test_parameter_description_as_dict_handled(self, documenter_config):
        """Parameter description as dict should be converted to string."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "parameters": [
                    {
                        "name": "config",
                        "type": {"base": "dict", "nested": "str"},  # Dict type
                        "description": {"what": "Configuration options"},  # Dict desc
                    }
                ],
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should convert to strings
        assert isinstance(result.documentation.parameters[0].type, str)
        assert isinstance(result.documentation.parameters[0].description, str)
        assert "Configuration" in result.documentation.parameters[0].description

    def test_raises_as_single_dict_handled(self, documenter_config):
        """raises as single dict instead of list should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "raises": {"type": "ValueError", "condition": "when input invalid"},
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should wrap in list
        assert len(result.documentation.raises) == 1
        assert result.documentation.raises[0].type == "ValueError"


class TestAdvancedJsonRepair:
    """Tests for advanced JSON repair scenarios."""

    def test_single_quotes_converted_to_double(self, documenter_config):
        """Single quotes in JSON should be converted to double quotes."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # JSON with single quotes (invalid but common LLM output)
        bad_json = "{'documentation': {'summary': 'Test', 'description': 'Desc'}, 'call_graph': {'callers': [], 'callees': []}}"

        result = agent._parse_response(bad_json, "test.Component", 100)

        assert result.documentation.summary == "Test"

    def test_javascript_comments_removed(self, documenter_config):
        """JavaScript-style comments should be removed from JSON."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # JSON with comments
        bad_json = """
        {
            // This is a comment
            "documentation": {
                "summary": "Test",
                "description": "Desc" /* inline comment */
            },
            "call_graph": {"callers": [], "callees": []}
        }
        """

        result = agent._parse_response(bad_json, "test.Component", 100)

        assert result.documentation.summary == "Test"

    def test_nan_infinity_handled(self, documenter_config):
        """NaN and Infinity in JSON should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # JSON with NaN/Infinity (invalid JSON but possible LLM output)
        bad_json = '{"documentation": {"summary": "Test", "description": "Desc"}, "call_graph": {"callers": [], "callees": []}, "confidence": NaN}'

        result = agent._parse_response(bad_json, "test.Component", 100)

        # Should parse with NaN converted to null -> default value
        assert result.documentation.summary == "Test"

    def test_unquoted_property_names_fixed(self, documenter_config):
        """Unquoted property names should be fixed."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # JSON with unquoted property names
        bad_json = '{documentation: {"summary": "Test", "description": "Desc"}, call_graph: {"callers": [], "callees": []}}'

        result = agent._parse_response(bad_json, "test.Component", 100)

        assert result.documentation.summary == "Test"

    def test_json_embedded_in_text_extracted(self, documenter_config):
        """JSON embedded in surrounding text should be extracted."""
        agent = ConcreteDocumenterAgent(documenter_config)

        # JSON embedded in text (common with some LLMs)
        bad_json = """
        Here is the documentation:
        {"documentation": {"summary": "Test", "description": "Desc"}, "call_graph": {"callers": [], "callees": []}}
        That's all!
        """

        result = agent._parse_response(bad_json, "test.Component", 100)

        assert result.documentation.summary == "Test"


class TestConfidenceStrings:
    """Tests for string-based confidence values."""

    def test_confidence_high_string(self, documenter_config):
        """Confidence 'high' should be converted to ~0.85."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {"callers": [], "callees": []},
            "confidence": "high",
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.metadata.confidence == 0.85

    def test_confidence_low_string(self, documenter_config):
        """Confidence 'low' should be converted to ~0.5."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {"callers": [], "callees": []},
            "confidence": "low",
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.metadata.confidence == 0.5

    def test_confidence_medium_string(self, documenter_config):
        """Confidence 'medium' should be converted to ~0.7."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {"callers": [], "callees": []},
            "confidence": "medium",
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.metadata.confidence == 0.7


class TestBooleanStrings:
    """Tests for string-based boolean values."""

    def test_required_true_string(self, documenter_config):
        """Parameter required='true' should be converted to True."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "parameters": [{"name": "param1", "required": "true"}],
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.documentation.parameters[0].required is True

    def test_required_false_string(self, documenter_config):
        """Parameter required='false' should be converted to False."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "parameters": [{"name": "param1", "required": "false"}],
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.documentation.parameters[0].required is False

    def test_required_yes_no_strings(self, documenter_config):
        """Parameter required='yes'/'no' should be converted to bool."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "parameters": [
                    {"name": "param1", "required": "yes"},
                    {"name": "param2", "required": "no"},
                ],
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.documentation.parameters[0].required is True
        assert result.documentation.parameters[1].required is False


class TestValidatorEdgeCases:
    """Tests for validator-specific edge cases."""

    def test_score_string_high(self, validator_config):
        """Score 'high' in validation should be converted correctly."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "passed",
            "completeness": {"score": "high"},
            "call_graph_accuracy": {"score": "medium"},
            "description_quality": {"score": "low"},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.completeness.score == 0.85
        assert result.call_graph_accuracy.score == 0.7
        assert result.description_quality.score == 0.5

    def test_missing_elements_as_string(self, validator_config):
        """missing_elements as string should be wrapped in list."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "warning",
            "completeness": {
                "score": 0.8,
                "missing_elements": "return type",  # String instead of list
            },
            "call_graph_accuracy": {"score": 1.0},
            "description_quality": {"score": 1.0},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert isinstance(result.completeness.missing_elements, list)
        assert "return type" in result.completeness.missing_elements

    def test_corrections_as_single_dict(self, validator_config):
        """corrections_applied as single dict should be wrapped in list."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "passed",
            "completeness": {"score": 1.0},
            "call_graph_accuracy": {"score": 1.0},
            "description_quality": {"score": 1.0},
            "corrections_applied": {
                "field": "description",
                "action": "modified",
                "reason": "Added more detail",
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert len(result.corrections_applied) == 1
        assert result.corrections_applied[0].field == "description"

    def test_validation_result_case_insensitive(self, validator_config):
        """validation_result should be case insensitive."""
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "PASSED",
            "completeness": {"score": 1.0},
            "call_graph_accuracy": {"score": 1.0},
            "description_quality": {"score": 1.0},
        })

        result = agent._parse_response(response, "test.Component", 100)

        from twinscribe.models.base import ValidationStatus
        assert result.validation_result == ValidationStatus.PASS

    def test_issues_as_list_of_dicts(self, validator_config):
        """description_quality.issues as list of dicts should be converted to strings.

        This is the exact error case from production:
        issues.0 Input should be a valid string [type=string_type,
        input_value={'issue_type': 'summary_r...this is a minor issue."}, input_type=dict]
        """
        agent = ConcreteValidatorAgent(validator_config)

        response = json.dumps({
            "validation_result": "warning",
            "completeness": {"score": 0.9},
            "call_graph_accuracy": {"score": 1.0},
            "description_quality": {
                "score": 0.8,
                "issues": [
                    {
                        "issue_type": "summary_redundant",
                        "severity": "minor",
                        "description": "Summary repeats the function name. This is a minor issue."
                    },
                    {
                        "issue_type": "missing_usage_context",
                        "severity": "minor",
                        "description": "Missing context about when to use this (LangChain LCEL graph)."
                    },
                ],
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        # Should convert dicts to strings and not crash
        assert isinstance(result.description_quality.issues, list)
        assert len(result.description_quality.issues) == 2
        # Each issue should be a string (converted from dict)
        assert all(isinstance(issue, str) for issue in result.description_quality.issues)
        # Content should be preserved
        assert "summary" in result.description_quality.issues[0].lower() or "redundant" in result.description_quality.issues[0].lower()


class TestCallerCalleeEdgeCases:
    """Tests for caller/callee edge cases."""

    def test_callers_as_string_list(self, documenter_config):
        """Callers as list of strings should be handled."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {
                "callers": ["foo.bar", "baz.qux"],  # Strings instead of dicts
                "callees": [],
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert len(result.call_graph.callers) == 2
        assert result.call_graph.callers[0].component_id == "foo.bar"
        assert result.call_graph.callers[1].component_id == "baz.qux"

    def test_component_id_as_number(self, documenter_config):
        """component_id as number should be converted to string."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {"summary": "Test", "description": "Desc"},
            "call_graph": {
                "callers": [{"component_id": 42}],  # Number instead of string
                "callees": [],
            },
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert len(result.call_graph.callers) == 1
        assert result.call_graph.callers[0].component_id == "42"

    def test_returns_as_list(self, documenter_config):
        """Returns as list should take first element."""
        agent = ConcreteDocumenterAgent(documenter_config)

        response = json.dumps({
            "documentation": {
                "summary": "Test",
                "description": "Desc",
                "returns": [
                    {"type": "str", "description": "The result"},
                    {"type": "int", "description": "The count"},
                ],
            },
            "call_graph": {"callers": [], "callees": []},
        })

        result = agent._parse_response(response, "test.Component", 100)

        assert result.documentation.returns is not None
        assert result.documentation.returns.type == "str"
        assert "result" in result.documentation.returns.description
