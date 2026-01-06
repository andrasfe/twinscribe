"""Unit tests for twinscribe.models.comparison module.

Tests cover:
- BeadsTicketRef model
- Discrepancy model
- ConvergenceStatus model
- ComparisonSummary model
- ComparatorMetadata model
- ComparisonResult model
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from twinscribe.models.base import DiscrepancyType, ResolutionAction
from twinscribe.models.comparison import (
    BeadsTicketRef,
    Discrepancy,
    ConvergenceStatus,
    ComparisonSummary,
    ComparatorMetadata,
    ComparisonResult,
)


class TestBeadsTicketRef:
    """Tests for BeadsTicketRef model."""

    def test_valid_ticket_ref(self):
        """Test creating a valid ticket reference."""
        ticket = BeadsTicketRef(
            summary="Missing documentation for exception",
            description="The ValueError exception is not documented in pkg.func",
            priority="High",
            ticket_key="LEGACY-123",
        )
        assert ticket.summary == "Missing documentation for exception"
        assert ticket.priority == "High"
        assert ticket.ticket_key == "LEGACY-123"

    def test_ticket_ref_without_key(self):
        """Test ticket ref before creation (no key yet)."""
        ticket = BeadsTicketRef(
            summary="Review required",
            description="Documentation disagreement needs review",
        )
        assert ticket.ticket_key is None
        assert ticket.priority == "Medium"  # Default


class TestDiscrepancy:
    """Tests for Discrepancy model."""

    def test_call_graph_discrepancy(self):
        """Test a call graph discrepancy."""
        disc = Discrepancy(
            discrepancy_id="disc_001",
            component_id="pkg.module.func",
            type=DiscrepancyType.CALL_GRAPH_EDGE,
            stream_a_value="pkg.utils.helper",
            stream_b_value=None,
            ground_truth="pkg.utils.helper",
            resolution=ResolutionAction.ACCEPT_GROUND_TRUTH,
            confidence=0.95,
        )
        assert disc.is_call_graph_related is True
        assert disc.is_resolved is True
        assert disc.is_blocking is False  # High confidence resolution

    def test_documentation_discrepancy(self):
        """Test a documentation content discrepancy."""
        disc = Discrepancy(
            discrepancy_id="disc_002",
            component_id="pkg.module.func",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            stream_a_value="Processes input data.",
            stream_b_value="Handles input processing.",
            resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
            requires_beads=True,
            beads_ticket=BeadsTicketRef(
                summary="Documentation wording disagreement",
                description="Streams A and B have different descriptions",
            ),
        )
        assert disc.is_call_graph_related is False
        assert disc.is_resolved is False
        assert disc.is_blocking is True
        assert disc.requires_beads is True

    def test_call_graph_related_types(self):
        """Test all call graph related types."""
        call_graph_types = [
            DiscrepancyType.CALL_GRAPH_EDGE,
            DiscrepancyType.CALL_SITE_LINE,
            DiscrepancyType.CALL_TYPE_MISMATCH,
        ]
        for disc_type in call_graph_types:
            disc = Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=disc_type,
            )
            assert disc.is_call_graph_related is True

    def test_documentation_related_types(self):
        """Test all documentation related types."""
        doc_types = [
            DiscrepancyType.DOCUMENTATION_CONTENT,
            DiscrepancyType.PARAMETER_DESCRIPTION,
            DiscrepancyType.RETURN_DESCRIPTION,
            DiscrepancyType.EXCEPTION_DOCUMENTATION,
            DiscrepancyType.MISSING_PARAMETER,
            DiscrepancyType.MISSING_EXCEPTION,
            DiscrepancyType.TYPE_ANNOTATION,
        ]
        for disc_type in doc_types:
            disc = Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=disc_type,
            )
            assert disc.is_call_graph_related is False

    def test_is_resolved_states(self):
        """Test is_resolved for different resolution actions."""
        resolved_actions = [
            ResolutionAction.ACCEPT_STREAM_A,
            ResolutionAction.ACCEPT_STREAM_B,
            ResolutionAction.ACCEPT_GROUND_TRUTH,
            ResolutionAction.MERGE_BOTH,
        ]
        for action in resolved_actions:
            disc = Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=DiscrepancyType.DOCUMENTATION_CONTENT,
                resolution=action,
            )
            assert disc.is_resolved is True

        unresolved_actions = [
            ResolutionAction.NEEDS_HUMAN_REVIEW,
            ResolutionAction.DEFERRED,
        ]
        for action in unresolved_actions:
            disc = Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=DiscrepancyType.DOCUMENTATION_CONTENT,
                resolution=action,
            )
            assert disc.is_resolved is False

    def test_is_blocking_with_low_confidence(self):
        """Test that low confidence resolved items are still not blocking.

        The is_blocking property returns True only for unresolved items.
        Resolved items (even with low confidence) are not blocking.
        High confidence resolutions get an explicit "not blocking" path.
        """
        disc = Discrepancy(
            discrepancy_id="test",
            component_id="test.func",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            resolution=ResolutionAction.ACCEPT_STREAM_A,
            confidence=0.5,  # Below 0.7 threshold
        )
        # Resolved items are not blocking, regardless of confidence
        assert disc.is_resolved is True
        assert disc.is_blocking is False

    def test_is_blocking_high_confidence_resolved(self):
        """Test that high confidence resolved items are explicitly not blocking."""
        disc = Discrepancy(
            discrepancy_id="test",
            component_id="test.func",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            resolution=ResolutionAction.ACCEPT_STREAM_A,
            confidence=0.9,  # Above 0.7 threshold
        )
        assert disc.is_resolved is True
        assert disc.is_blocking is False

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=DiscrepancyType.DOCUMENTATION_CONTENT,
                confidence=-0.1,
            )
        with pytest.raises(ValidationError):
            Discrepancy(
                discrepancy_id="test",
                component_id="test.func",
                type=DiscrepancyType.DOCUMENTATION_CONTENT,
                confidence=1.1,
            )

    def test_iteration_tracking(self):
        """Test iteration_found tracking."""
        disc = Discrepancy(
            discrepancy_id="test",
            component_id="test.func",
            type=DiscrepancyType.DOCUMENTATION_CONTENT,
            iteration_found=3,
        )
        assert disc.iteration_found == 3


class TestConvergenceStatus:
    """Tests for ConvergenceStatus model."""

    def test_converged_status(self):
        """Test converged status."""
        status = ConvergenceStatus(
            converged=True,
            blocking_discrepancies=0,
            recommendation="finalize",
        )
        assert status.converged is True
        assert status.blocking_discrepancies == 0

    def test_not_converged_status(self):
        """Test not converged status."""
        status = ConvergenceStatus(
            converged=False,
            blocking_discrepancies=5,
            recommendation="continue",
        )
        assert status.converged is False
        assert status.blocking_discrepancies == 5

    def test_default_values(self):
        """Test default values."""
        status = ConvergenceStatus()
        assert status.converged is False
        assert status.blocking_discrepancies == 0
        assert status.recommendation == "continue"


class TestComparisonSummary:
    """Tests for ComparisonSummary model."""

    def test_perfect_agreement(self):
        """Test summary with perfect agreement."""
        summary = ComparisonSummary(
            total_components=10,
            identical=10,
            discrepancies=0,
        )
        assert summary.agreement_rate == 1.0

    def test_partial_agreement(self):
        """Test summary with partial agreement."""
        summary = ComparisonSummary(
            total_components=10,
            identical=7,
            discrepancies=5,
            resolved_by_ground_truth=3,
            requires_human_review=2,
        )
        assert summary.agreement_rate == 0.7
        assert summary.resolved_by_ground_truth == 3
        assert summary.requires_human_review == 2

    def test_no_components(self):
        """Test agreement_rate with no components."""
        summary = ComparisonSummary(total_components=0)
        assert summary.agreement_rate == 1.0  # Default to 1.0 for empty

    def test_default_values(self):
        """Test default values."""
        summary = ComparisonSummary()
        assert summary.total_components == 0
        assert summary.identical == 0
        assert summary.discrepancies == 0


class TestComparatorMetadata:
    """Tests for ComparatorMetadata model."""

    def test_valid_metadata(self):
        """Test creating valid comparator metadata."""
        meta = ComparatorMetadata(
            model="claude-opus-4-5",
            comparison_duration_ms=1500,
            token_count=3000,
        )
        assert meta.agent_id == "C"
        assert meta.model == "claude-opus-4-5"
        assert meta.comparison_duration_ms == 1500
        assert meta.token_count == 3000

    def test_timestamp_defaults(self):
        """Test timestamp defaults to now."""
        before = datetime.utcnow()
        meta = ComparatorMetadata(model="test")
        after = datetime.utcnow()
        assert before <= meta.timestamp <= after


class TestComparisonResult:
    """Tests for ComparisonResult model."""

    @pytest.fixture
    def converged_result(self):
        """Create a converged comparison result."""
        return ComparisonResult(
            comparison_id="cmp_001",
            iteration=3,
            summary=ComparisonSummary(
                total_components=10,
                identical=10,
                discrepancies=0,
            ),
            convergence_status=ConvergenceStatus(
                converged=True,
                blocking_discrepancies=0,
                recommendation="finalize",
            ),
            metadata=ComparatorMetadata(model="claude-opus-4"),
        )

    @pytest.fixture
    def not_converged_result(self):
        """Create a not-converged comparison result."""
        return ComparisonResult(
            comparison_id="cmp_002",
            iteration=1,
            summary=ComparisonSummary(
                total_components=10,
                identical=5,
                discrepancies=8,
                resolved_by_ground_truth=3,
                requires_human_review=2,
            ),
            discrepancies=[
                Discrepancy(
                    discrepancy_id="disc_001",
                    component_id="pkg.func1",
                    type=DiscrepancyType.CALL_GRAPH_EDGE,
                    resolution=ResolutionAction.ACCEPT_GROUND_TRUTH,
                    confidence=0.95,
                ),
                Discrepancy(
                    discrepancy_id="disc_002",
                    component_id="pkg.func2",
                    type=DiscrepancyType.DOCUMENTATION_CONTENT,
                    resolution=ResolutionAction.NEEDS_HUMAN_REVIEW,
                    requires_beads=True,
                ),
            ],
            convergence_status=ConvergenceStatus(
                converged=False,
                blocking_discrepancies=1,
                recommendation="continue",
            ),
            metadata=ComparatorMetadata(model="claude-opus-4"),
        )

    def test_converged_result(self, converged_result):
        """Test properties of converged result."""
        assert converged_result.is_converged is True
        assert converged_result.summary.agreement_rate == 1.0

    def test_not_converged_result(self, not_converged_result):
        """Test properties of not-converged result."""
        assert not_converged_result.is_converged is False
        assert len(not_converged_result.discrepancies) == 2

    def test_get_discrepancies_for_component(self, not_converged_result):
        """Test filtering discrepancies by component."""
        discs = not_converged_result.get_discrepancies_for_component("pkg.func1")
        assert len(discs) == 1
        assert discs[0].discrepancy_id == "disc_001"

    def test_get_blocking_discrepancies(self, not_converged_result):
        """Test getting blocking discrepancies."""
        blocking = not_converged_result.get_blocking_discrepancies()
        assert len(blocking) == 1
        assert blocking[0].discrepancy_id == "disc_002"

    def test_get_beads_required(self, not_converged_result):
        """Test getting discrepancies requiring Beads tickets."""
        beads = not_converged_result.get_beads_required()
        assert len(beads) == 1
        assert beads[0].requires_beads is True

    def test_iteration_tracking(self):
        """Test iteration field validation."""
        result = ComparisonResult(
            comparison_id="test",
            iteration=5,
            metadata=ComparatorMetadata(model="test"),
        )
        assert result.iteration == 5

        with pytest.raises(ValidationError):
            ComparisonResult(
                comparison_id="test",
                iteration=0,  # Must be >= 1
                metadata=ComparatorMetadata(model="test"),
            )

    def test_json_serialization(self, not_converged_result):
        """Test JSON roundtrip."""
        json_str = not_converged_result.model_dump_json()
        restored = ComparisonResult.model_validate_json(json_str)

        assert restored.comparison_id == not_converged_result.comparison_id
        assert restored.is_converged == not_converged_result.is_converged
        assert len(restored.discrepancies) == len(not_converged_result.discrepancies)
