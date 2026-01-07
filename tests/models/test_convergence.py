"""Unit tests for twinscribe.models.convergence module.

Tests cover:
- ConvergenceCriteria model and is_satisfied method
- ConvergenceHistoryEntry model
- ConvergenceReport model and helper methods
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from twinscribe.models.convergence import (
    ConvergenceCriteria,
    ConvergenceHistoryEntry,
    ConvergenceReport,
)


class TestConvergenceCriteria:
    """Tests for ConvergenceCriteria model."""

    def test_default_criteria(self):
        """Test default convergence criteria."""
        criteria = ConvergenceCriteria()
        assert criteria.max_iterations == 5
        assert criteria.call_graph_match_rate == 0.98
        assert criteria.documentation_similarity == 0.95
        assert criteria.max_open_discrepancies == 2
        assert len(criteria.blocking_discrepancy_types) > 0

    def test_custom_criteria(self):
        """Test custom convergence criteria."""
        criteria = ConvergenceCriteria(
            max_iterations=10,
            call_graph_match_rate=0.99,
            documentation_similarity=0.90,
            max_open_discrepancies=5,
            blocking_discrepancy_types=["critical_error"],
        )
        assert criteria.max_iterations == 10
        assert criteria.call_graph_match_rate == 0.99
        assert criteria.max_open_discrepancies == 5

    def test_max_iterations_bounds(self):
        """Test max_iterations must be 1-20."""
        # Valid bounds
        criteria_min = ConvergenceCriteria(max_iterations=1)
        criteria_max = ConvergenceCriteria(max_iterations=20)
        assert criteria_min.max_iterations == 1
        assert criteria_max.max_iterations == 20

        # Invalid bounds
        with pytest.raises(ValidationError):
            ConvergenceCriteria(max_iterations=0)
        with pytest.raises(ValidationError):
            ConvergenceCriteria(max_iterations=21)

    def test_rate_bounds(self):
        """Test rate fields must be 0.0-1.0."""
        # Valid
        criteria = ConvergenceCriteria(
            call_graph_match_rate=0.0,
            documentation_similarity=1.0,
        )
        assert criteria.call_graph_match_rate == 0.0
        assert criteria.documentation_similarity == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            ConvergenceCriteria(call_graph_match_rate=1.1)
        with pytest.raises(ValidationError):
            ConvergenceCriteria(documentation_similarity=-0.1)

    def test_is_satisfied_all_criteria_met(self):
        """Test is_satisfied when all criteria are met."""
        criteria = ConvergenceCriteria(
            call_graph_match_rate=0.95,
            documentation_similarity=0.90,
            max_open_discrepancies=3,
        )
        result = criteria.is_satisfied(
            call_graph_match=0.98,
            doc_similarity=0.95,
            open_discrepancies=2,
            has_blocking=False,
        )
        assert result is True

    def test_is_satisfied_call_graph_below_threshold(self):
        """Test is_satisfied when call graph match is below threshold."""
        criteria = ConvergenceCriteria(call_graph_match_rate=0.98)
        result = criteria.is_satisfied(
            call_graph_match=0.90,  # Below 0.98
            doc_similarity=1.0,
            open_discrepancies=0,
            has_blocking=False,
        )
        assert result is False

    def test_is_satisfied_doc_similarity_below_threshold(self):
        """Test is_satisfied when doc similarity is below threshold."""
        criteria = ConvergenceCriteria(documentation_similarity=0.95)
        result = criteria.is_satisfied(
            call_graph_match=1.0,
            doc_similarity=0.80,  # Below 0.95
            open_discrepancies=0,
            has_blocking=False,
        )
        assert result is False

    def test_is_satisfied_too_many_open_discrepancies(self):
        """Test is_satisfied when too many open discrepancies."""
        criteria = ConvergenceCriteria(max_open_discrepancies=2)
        result = criteria.is_satisfied(
            call_graph_match=1.0,
            doc_similarity=1.0,
            open_discrepancies=5,  # Above 2
            has_blocking=False,
        )
        assert result is False

    def test_is_satisfied_with_blocking(self):
        """Test is_satisfied when blocking discrepancies exist."""
        criteria = ConvergenceCriteria()
        result = criteria.is_satisfied(
            call_graph_match=1.0,
            doc_similarity=1.0,
            open_discrepancies=0,
            has_blocking=True,  # Blocking present
        )
        assert result is False

    def test_is_satisfied_exact_thresholds(self):
        """Test is_satisfied at exact threshold values."""
        criteria = ConvergenceCriteria(
            call_graph_match_rate=0.98,
            documentation_similarity=0.95,
            max_open_discrepancies=2,
        )
        result = criteria.is_satisfied(
            call_graph_match=0.98,  # Exactly at threshold
            doc_similarity=0.95,  # Exactly at threshold
            open_discrepancies=2,  # Exactly at max
            has_blocking=False,
        )
        assert result is True


class TestConvergenceHistoryEntry:
    """Tests for ConvergenceHistoryEntry model."""

    def test_valid_entry(self):
        """Test creating a valid history entry."""
        entry = ConvergenceHistoryEntry(
            iteration=1,
            total_components=50,
            identical=45,
            discrepancies=10,
            resolved=8,
            blocking=2,
            call_graph_match_rate=0.95,
            documentation_similarity=0.92,
            beads_tickets_created=1,
        )
        assert entry.iteration == 1
        assert entry.total_components == 50
        assert entry.agreement_rate == 0.9  # 45/50

    def test_minimal_entry(self):
        """Test entry with only required fields."""
        entry = ConvergenceHistoryEntry(iteration=1)
        assert entry.total_components == 0
        assert entry.identical == 0
        assert entry.discrepancies == 0
        assert entry.resolved == 0
        assert entry.blocking == 0
        assert entry.call_graph_match_rate == 0.0
        assert entry.documentation_similarity == 0.0

    def test_iteration_must_be_positive(self):
        """Test iteration must be >= 1."""
        with pytest.raises(ValidationError):
            ConvergenceHistoryEntry(iteration=0)

    def test_agreement_rate_calculation(self):
        """Test agreement_rate property."""
        entry = ConvergenceHistoryEntry(
            iteration=1,
            total_components=100,
            identical=75,
        )
        assert entry.agreement_rate == 0.75

    def test_agreement_rate_zero_components(self):
        """Test agreement_rate with zero components."""
        entry = ConvergenceHistoryEntry(iteration=1, total_components=0)
        assert entry.agreement_rate == 1.0  # Default to 1.0

    def test_timestamp_defaults_to_now(self):
        """Test timestamp defaults to current time."""
        before = datetime.utcnow()
        entry = ConvergenceHistoryEntry(iteration=1)
        after = datetime.utcnow()
        assert before <= entry.timestamp <= after

    def test_json_serialization(self):
        """Test JSON roundtrip."""
        entry = ConvergenceHistoryEntry(
            iteration=3,
            total_components=100,
            identical=90,
            discrepancies=15,
            resolved=10,
            blocking=1,
            call_graph_match_rate=0.98,
            documentation_similarity=0.96,
        )
        json_str = entry.model_dump_json()
        restored = ConvergenceHistoryEntry.model_validate_json(json_str)
        assert restored.iteration == entry.iteration
        assert restored.agreement_rate == entry.agreement_rate


class TestConvergenceReport:
    """Tests for ConvergenceReport model."""

    @pytest.fixture
    def successful_report(self):
        """Create a successful convergence report."""
        start = datetime.utcnow()
        end = start + timedelta(minutes=5)
        return ConvergenceReport(
            total_iterations=3,
            final_status="converged",
            history=[
                ConvergenceHistoryEntry(
                    iteration=1,
                    total_components=50,
                    identical=30,
                    discrepancies=20,
                ),
                ConvergenceHistoryEntry(
                    iteration=2,
                    total_components=50,
                    identical=45,
                    discrepancies=5,
                ),
                ConvergenceHistoryEntry(
                    iteration=3,
                    total_components=50,
                    identical=50,
                    discrepancies=0,
                ),
            ],
            started_at=start,
            completed_at=end,
            forced_convergence=False,
        )

    @pytest.fixture
    def forced_report(self):
        """Create a forced convergence report."""
        return ConvergenceReport(
            total_iterations=5,
            final_status="max_iterations_reached",
            forced_convergence=True,
            remaining_discrepancies=["disc_001", "disc_002"],
        )

    def test_successful_report(self, successful_report):
        """Test properties of successful report."""
        assert successful_report.is_successful is True
        assert successful_report.final_status == "converged"
        assert successful_report.forced_convergence is False
        assert successful_report.duration_seconds is not None
        assert successful_report.duration_seconds >= 0

    def test_forced_report(self, forced_report):
        """Test properties of forced convergence report."""
        assert forced_report.is_successful is False
        assert forced_report.forced_convergence is True
        assert len(forced_report.remaining_discrepancies) == 2

    def test_empty_report(self):
        """Test empty report defaults."""
        report = ConvergenceReport()
        assert report.total_iterations == 0
        assert report.final_status == "pending"
        assert report.history == []
        assert report.forced_convergence is False
        assert report.remaining_discrepancies == []
        assert report.duration_seconds is None

    def test_duration_seconds_calculation(self):
        """Test duration calculation."""
        start = datetime(2026, 1, 1, 10, 0, 0)
        end = datetime(2026, 1, 1, 10, 5, 30)  # 5.5 minutes later
        report = ConvergenceReport(
            started_at=start,
            completed_at=end,
        )
        assert report.duration_seconds == 330.0  # 5*60 + 30

    def test_duration_seconds_none_when_incomplete(self):
        """Test duration is None when timestamps missing."""
        report = ConvergenceReport(started_at=datetime.utcnow())
        assert report.duration_seconds is None

        report2 = ConvergenceReport(completed_at=datetime.utcnow())
        assert report2.duration_seconds is None

    def test_add_iteration(self):
        """Test adding iteration entries."""
        report = ConvergenceReport()
        assert report.total_iterations == 0

        entry1 = ConvergenceHistoryEntry(iteration=1, total_components=50)
        report.add_iteration(entry1)
        assert report.total_iterations == 1
        assert len(report.history) == 1

        entry2 = ConvergenceHistoryEntry(iteration=2, total_components=50)
        report.add_iteration(entry2)
        assert report.total_iterations == 2

    def test_get_latest_entry(self, successful_report):
        """Test getting latest entry."""
        latest = successful_report.get_latest_entry()
        assert latest is not None
        assert latest.iteration == 3
        assert latest.identical == 50

    def test_get_latest_entry_empty(self):
        """Test getting latest entry from empty report."""
        report = ConvergenceReport()
        assert report.get_latest_entry() is None

    def test_is_successful_requires_both_conditions(self):
        """Test is_successful requires converged status AND not forced."""
        # Converged but forced
        report1 = ConvergenceReport(
            final_status="converged",
            forced_convergence=True,
        )
        assert report1.is_successful is False

        # Not converged and not forced
        report2 = ConvergenceReport(
            final_status="pending",
            forced_convergence=False,
        )
        assert report2.is_successful is False

        # Converged and not forced
        report3 = ConvergenceReport(
            final_status="converged",
            forced_convergence=False,
        )
        assert report3.is_successful is True

    def test_default_criteria(self):
        """Test default criteria is applied."""
        report = ConvergenceReport()
        assert report.criteria.max_iterations == 5
        assert report.criteria.call_graph_match_rate == 0.98

    def test_json_serialization(self, successful_report):
        """Test JSON roundtrip."""
        json_str = successful_report.model_dump_json()
        restored = ConvergenceReport.model_validate_json(json_str)

        assert restored.total_iterations == successful_report.total_iterations
        assert restored.final_status == successful_report.final_status
        assert len(restored.history) == len(successful_report.history)
        assert restored.is_successful == successful_report.is_successful
