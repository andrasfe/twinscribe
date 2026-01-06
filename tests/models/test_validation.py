"""Unit tests for twinscribe.models.validation module.

Tests cover:
- CompletenessCheck model
- CallGraphAccuracy model
- CorrectionApplied model
- ValidatorMetadata model
- ValidationResult model
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from twinscribe.models.base import StreamId, ValidationStatus
from twinscribe.models.validation import (
    CompletenessCheck,
    CallGraphAccuracy,
    CorrectionApplied,
    ValidatorMetadata,
    ValidationResult,
)


class TestCompletenessCheck:
    """Tests for CompletenessCheck model."""

    def test_complete_check(self):
        """Test a complete check with no issues."""
        check = CompletenessCheck(score=1.0)
        assert check.score == 1.0
        assert check.missing_elements == []
        assert check.extra_elements == []
        assert check.is_complete is True

    def test_incomplete_check(self):
        """Test a check with missing elements."""
        check = CompletenessCheck(
            score=0.75,
            missing_elements=[
                "exception: ValueError not documented",
                "parameter 'timeout' missing description",
            ],
        )
        assert check.score == 0.75
        assert len(check.missing_elements) == 2
        assert check.is_complete is False

    def test_check_with_extra_elements(self):
        """Test a check with extra documented elements."""
        check = CompletenessCheck(
            score=0.9,
            extra_elements=["parameter 'debug' not in signature"],
        )
        assert len(check.extra_elements) == 1
        # Still complete if no missing elements
        assert check.is_complete is True

    def test_score_bounds(self):
        """Test score must be between 0 and 1."""
        # Valid bounds
        check_min = CompletenessCheck(score=0.0)
        check_max = CompletenessCheck(score=1.0)
        assert check_min.score == 0.0
        assert check_max.score == 1.0

        # Invalid bounds
        with pytest.raises(ValidationError):
            CompletenessCheck(score=-0.1)
        with pytest.raises(ValidationError):
            CompletenessCheck(score=1.1)


class TestCallGraphAccuracy:
    """Tests for CallGraphAccuracy model."""

    def test_perfect_accuracy(self):
        """Test perfect accuracy with all edges verified."""
        accuracy = CallGraphAccuracy(
            score=1.0,
            verified_callees=["func_a", "func_b"],
            verified_callers=["main"],
        )
        assert accuracy.score == 1.0
        assert len(accuracy.verified_callees) == 2
        assert accuracy.has_errors is False

    def test_has_errors_with_false_callees(self):
        """Test has_errors when false callees exist."""
        accuracy = CallGraphAccuracy(
            score=0.8,
            verified_callees=["func_a"],
            false_callees=["nonexistent"],
        )
        assert accuracy.has_errors is True

    def test_has_errors_with_missing_callees(self):
        """Test has_errors when callees are missing."""
        accuracy = CallGraphAccuracy(
            score=0.5,
            verified_callees=["func_a"],
            missing_callees=["func_b", "func_c"],
        )
        assert accuracy.has_errors is True

    def test_callee_precision(self):
        """Test callee precision calculation."""
        accuracy = CallGraphAccuracy(
            verified_callees=["a", "b", "c"],  # 3 correct
            false_callees=["x"],  # 1 false
        )
        assert accuracy.callee_precision == 0.75  # 3/(3+1)

    def test_callee_precision_no_callees(self):
        """Test precision is 1.0 when no callees documented."""
        accuracy = CallGraphAccuracy()
        assert accuracy.callee_precision == 1.0

    def test_callee_recall(self):
        """Test callee recall calculation."""
        accuracy = CallGraphAccuracy(
            verified_callees=["a", "b"],  # 2 found
            missing_callees=["c", "d"],  # 2 missing
        )
        assert accuracy.callee_recall == 0.5  # 2/(2+2)

    def test_callee_recall_no_truth(self):
        """Test recall is 1.0 when no callees in truth."""
        accuracy = CallGraphAccuracy()
        assert accuracy.callee_recall == 1.0


class TestCorrectionApplied:
    """Tests for CorrectionApplied model."""

    def test_valid_correction(self):
        """Test creating a valid correction record."""
        correction = CorrectionApplied(
            field="call_graph.callees",
            action="removed",
            original_value="pkg.nonexistent",
            corrected_value=None,
            reason="Not found in static analysis",
        )
        assert correction.field == "call_graph.callees"
        assert correction.action == "removed"
        assert correction.reason == "Not found in static analysis"

    def test_add_correction(self):
        """Test an add correction."""
        correction = CorrectionApplied(
            field="call_graph.callees",
            action="added",
            original_value=None,
            corrected_value="pkg.missing_func",
            reason="Found in static analysis",
        )
        assert correction.action == "added"
        assert correction.corrected_value == "pkg.missing_func"

    def test_modification_correction(self):
        """Test a modification correction."""
        correction = CorrectionApplied(
            field="documentation.parameters[0].type",
            action="modified",
            original_value="str",
            corrected_value="Optional[str]",
            reason="Signature indicates optional parameter",
        )
        assert correction.action == "modified"


class TestValidatorMetadata:
    """Tests for ValidatorMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid validator metadata."""
        meta = ValidatorMetadata(
            agent_id="A2",
            stream_id=StreamId.STREAM_A,
            model="claude-haiku-4-5",
            static_analyzer="pycg",
            token_count=500,
        )
        assert meta.agent_id == "A2"
        assert meta.stream_id == StreamId.STREAM_A
        assert meta.static_analyzer == "pycg"
        assert meta.token_count == 500

    def test_timestamp_defaults(self):
        """Test timestamp defaults to now."""
        before = datetime.utcnow()
        meta = ValidatorMetadata(
            agent_id="B2",
            stream_id=StreamId.STREAM_B,
            model="gpt-4o-mini",
        )
        after = datetime.utcnow()
        assert before <= meta.timestamp <= after

    def test_static_analyzer_default(self):
        """Test static_analyzer defaults to pycg."""
        meta = ValidatorMetadata(
            agent_id="A2",
            stream_id=StreamId.STREAM_A,
            model="test",
        )
        assert meta.static_analyzer == "pycg"


class TestValidationResult:
    """Tests for ValidationResult model."""

    @pytest.fixture
    def passing_result(self):
        """Create a passing validation result."""
        return ValidationResult(
            component_id="pkg.module.func",
            validation_result=ValidationStatus.PASS,
            completeness=CompletenessCheck(score=1.0),
            call_graph_accuracy=CallGraphAccuracy(
                score=1.0,
                verified_callees=["pkg.utils"],
            ),
            metadata=ValidatorMetadata(
                agent_id="A2",
                stream_id=StreamId.STREAM_A,
                model="claude-haiku",
            ),
        )

    @pytest.fixture
    def failing_result(self):
        """Create a failing validation result."""
        return ValidationResult(
            component_id="pkg.module.bad_func",
            validation_result=ValidationStatus.FAIL,
            completeness=CompletenessCheck(
                score=0.5,
                missing_elements=["exception not documented"],
            ),
            call_graph_accuracy=CallGraphAccuracy(
                score=0.6,
                false_callees=["nonexistent"],
            ),
            corrections_applied=[
                CorrectionApplied(
                    field="call_graph.callees",
                    action="removed",
                    original_value="nonexistent",
                    reason="Not in static analysis",
                ),
            ],
            metadata=ValidatorMetadata(
                agent_id="A2",
                stream_id=StreamId.STREAM_A,
                model="claude-haiku",
            ),
        )

    def test_passing_result(self, passing_result):
        """Test properties of passing result."""
        assert passing_result.is_valid is True
        assert passing_result.requires_iteration is False
        assert passing_result.total_corrections == 0

    def test_failing_result(self, failing_result):
        """Test properties of failing result."""
        assert failing_result.is_valid is False
        assert failing_result.requires_iteration is True
        assert failing_result.total_corrections == 1

    def test_warning_result(self):
        """Test warning status is still valid."""
        result = ValidationResult(
            component_id="pkg.func",
            validation_result=ValidationStatus.WARNING,
            metadata=ValidatorMetadata(
                agent_id="A2",
                stream_id=StreamId.STREAM_A,
                model="test",
            ),
        )
        assert result.is_valid is True

    def test_requires_iteration_with_corrections(self):
        """Test requires_iteration is true when corrections applied."""
        result = ValidationResult(
            component_id="pkg.func",
            validation_result=ValidationStatus.PASS,  # Passed after correction
            corrections_applied=[
                CorrectionApplied(field="test", action="modified"),
            ],
            metadata=ValidatorMetadata(
                agent_id="A2",
                stream_id=StreamId.STREAM_A,
                model="test",
            ),
        )
        assert result.requires_iteration is True

    def test_empty_component_id_rejected(self):
        """Test empty component_id is rejected."""
        with pytest.raises(ValidationError):
            ValidationResult(
                component_id="",
                validation_result=ValidationStatus.PASS,
                metadata=ValidatorMetadata(
                    agent_id="A2",
                    stream_id=StreamId.STREAM_A,
                    model="test",
                ),
            )

    def test_json_serialization(self, failing_result):
        """Test JSON roundtrip."""
        json_str = failing_result.model_dump_json()
        restored = ValidationResult.model_validate_json(json_str)

        assert restored.component_id == failing_result.component_id
        assert restored.validation_result == ValidationStatus.FAIL
        assert len(restored.corrections_applied) == 1
        assert restored.completeness.score == 0.5
