"""
Convergence Management.

Handles convergence criteria, checking, and reporting for the dual-stream
documentation system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from twinscribe.models.comparison import ComparisonResult, Discrepancy


class ConvergenceStatus(str, Enum):
    """Status of convergence."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    CONVERGED = "converged"
    PARTIALLY_CONVERGED = "partially_converged"
    FORCED = "forced"  # Max iterations reached
    FAILED = "failed"


class BlockingDiscrepancyType(str, Enum):
    """Types of discrepancies that block convergence."""

    MISSING_CRITICAL_CALL = "missing_critical_call"
    FALSE_CRITICAL_CALL = "false_critical_call"
    MISSING_PUBLIC_API_DOC = "missing_public_api_doc"
    SECURITY_RELEVANT_GAP = "security_relevant_gap"
    TYPE_MISMATCH = "type_mismatch"
    SIGNATURE_MISMATCH = "signature_mismatch"


class ConvergenceCriteria(BaseModel):
    """Criteria for determining convergence.

    Attributes:
        max_iterations: Hard limit on iterations
        call_graph_match_threshold: Required call graph match rate
        documentation_similarity_threshold: Required doc similarity
        max_open_discrepancies: Max unresolved non-blocking issues
        beads_ticket_timeout_hours: Timeout for Beads resolution
        blocking_types: Discrepancy types that block convergence
        require_static_validation: Require static analysis validation
    """

    max_iterations: int = Field(default=5, ge=1, le=20)
    call_graph_match_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
    documentation_similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_open_discrepancies: int = Field(default=2, ge=0)
    beads_ticket_timeout_hours: int = Field(default=48, ge=0)
    blocking_types: list[BlockingDiscrepancyType] = Field(
        default_factory=lambda: [
            BlockingDiscrepancyType.MISSING_CRITICAL_CALL,
            BlockingDiscrepancyType.FALSE_CRITICAL_CALL,
            BlockingDiscrepancyType.MISSING_PUBLIC_API_DOC,
            BlockingDiscrepancyType.SECURITY_RELEVANT_GAP,
        ]
    )
    require_static_validation: bool = Field(default=True)


@dataclass
class ConvergenceCheck:
    """Result of a convergence check.

    Attributes:
        converged: Whether convergence is achieved
        status: Convergence status
        call_graph_match_rate: Current call graph match rate
        documentation_similarity: Current documentation similarity
        blocking_count: Number of blocking discrepancies
        non_blocking_count: Number of non-blocking discrepancies
        details: Additional check details
    """

    converged: bool
    status: ConvergenceStatus
    call_graph_match_rate: float
    documentation_similarity: float
    blocking_count: int
    non_blocking_count: int
    details: dict[str, Any] = field(default_factory=dict)


class ConvergenceChecker:
    """Checks convergence criteria against comparison results.

    Evaluates:
    - Call graph match rates against threshold
    - Documentation similarity against threshold
    - Blocking discrepancy presence
    - Open discrepancy count

    Usage:
        checker = ConvergenceChecker(criteria)
        result = checker.check(comparison_result)
        if result.converged:
            print("Streams have converged!")
    """

    def __init__(self, criteria: ConvergenceCriteria) -> None:
        """Initialize the checker.

        Args:
            criteria: Convergence criteria to apply
        """
        self._criteria = criteria

    @property
    def criteria(self) -> ConvergenceCriteria:
        """Get convergence criteria."""
        return self._criteria

    def check(
        self,
        comparison: "ComparisonResult",
        iteration: int,
    ) -> ConvergenceCheck:
        """Check convergence against comparison result.

        Args:
            comparison: Latest comparison result
            iteration: Current iteration number

        Returns:
            Convergence check result
        """
        # Extract metrics from comparison
        summary = comparison.summary
        call_graph_match_rate = summary.call_graph_match_rate
        documentation_similarity = summary.documentation_similarity

        # Count discrepancies by blocking status
        blocking = []
        non_blocking = []

        for disc in comparison.discrepancies:
            if self._is_blocking(disc):
                blocking.append(disc)
            else:
                non_blocking.append(disc)

        # Check each criterion
        checks = {}

        # Call graph threshold
        checks["call_graph"] = call_graph_match_rate >= self._criteria.call_graph_match_threshold

        # Documentation similarity threshold
        checks["documentation"] = (
            documentation_similarity >= self._criteria.documentation_similarity_threshold
        )

        # No blocking discrepancies
        checks["no_blocking"] = len(blocking) == 0

        # Open discrepancies within limit
        checks["discrepancy_limit"] = len(non_blocking) <= self._criteria.max_open_discrepancies

        # Determine overall convergence
        converged = all(checks.values())

        # Determine status
        if converged:
            status = ConvergenceStatus.CONVERGED
        elif iteration >= self._criteria.max_iterations:
            status = ConvergenceStatus.FORCED
        elif any(checks.values()):
            status = ConvergenceStatus.PARTIALLY_CONVERGED
        else:
            status = ConvergenceStatus.IN_PROGRESS

        return ConvergenceCheck(
            converged=converged,
            status=status,
            call_graph_match_rate=call_graph_match_rate,
            documentation_similarity=documentation_similarity,
            blocking_count=len(blocking),
            non_blocking_count=len(non_blocking),
            details={
                "checks": checks,
                "blocking_discrepancies": [d.id for d in blocking],
                "iteration": iteration,
                "max_iterations": self._criteria.max_iterations,
            },
        )

    def _is_blocking(self, discrepancy: "Discrepancy") -> bool:
        """Check if a discrepancy is blocking.

        Args:
            discrepancy: Discrepancy to check

        Returns:
            True if blocking
        """
        # Check against blocking types
        for blocking_type in self._criteria.blocking_types:
            if discrepancy.type.value == blocking_type.value:
                return True

        # Check explicit blocking flag
        if hasattr(discrepancy, "is_blocking") and discrepancy.is_blocking:
            return True

        return False

    def get_required_actions(
        self,
        check: ConvergenceCheck,
    ) -> list[str]:
        """Get required actions to achieve convergence.

        Args:
            check: Convergence check result

        Returns:
            List of required actions
        """
        actions = []

        details = check.details.get("checks", {})

        if not details.get("call_graph", True):
            gap = self._criteria.call_graph_match_threshold - check.call_graph_match_rate
            actions.append(
                f"Improve call graph match rate by {gap:.1%} "
                f"(current: {check.call_graph_match_rate:.1%}, "
                f"required: {self._criteria.call_graph_match_threshold:.1%})"
            )

        if not details.get("documentation", True):
            gap = self._criteria.documentation_similarity_threshold - check.documentation_similarity
            actions.append(
                f"Improve documentation similarity by {gap:.1%} "
                f"(current: {check.documentation_similarity:.1%}, "
                f"required: {self._criteria.documentation_similarity_threshold:.1%})"
            )

        if not details.get("no_blocking", True):
            actions.append(f"Resolve {check.blocking_count} blocking discrepancies")

        if not details.get("discrepancy_limit", True):
            excess = check.non_blocking_count - self._criteria.max_open_discrepancies
            actions.append(
                f"Resolve {excess} non-blocking discrepancies "
                f"(current: {check.non_blocking_count}, "
                f"max: {self._criteria.max_open_discrepancies})"
            )

        return actions


@dataclass
class ConvergenceHistoryEntry:
    """Entry in convergence history.

    Attributes:
        iteration: Iteration number
        timestamp: When check was performed
        check_result: Convergence check result
        actions_taken: Actions taken in this iteration
    """

    iteration: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    check_result: ConvergenceCheck | None = None
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "check_result": {
                "converged": self.check_result.converged,
                "status": self.check_result.status.value,
                "call_graph_match_rate": self.check_result.call_graph_match_rate,
                "documentation_similarity": self.check_result.documentation_similarity,
                "blocking_count": self.check_result.blocking_count,
                "non_blocking_count": self.check_result.non_blocking_count,
            }
            if self.check_result
            else None,
            "actions_taken": self.actions_taken,
        }


class ConvergenceTracker:
    """Tracks convergence progress over iterations.

    Provides:
    - History of convergence checks
    - Trend analysis
    - Recommendations

    Usage:
        tracker = ConvergenceTracker(checker)

        # After each iteration
        tracker.record(iteration, check_result, actions)

        # Get recommendations
        recommendations = tracker.get_recommendations()
    """

    def __init__(self, checker: ConvergenceChecker) -> None:
        """Initialize the tracker.

        Args:
            checker: Convergence checker instance
        """
        self._checker = checker
        self._history: list[ConvergenceHistoryEntry] = []

    @property
    def history(self) -> list[ConvergenceHistoryEntry]:
        """Get convergence history."""
        return self._history

    @property
    def latest(self) -> ConvergenceHistoryEntry | None:
        """Get latest history entry."""
        return self._history[-1] if self._history else None

    def record(
        self,
        iteration: int,
        check_result: ConvergenceCheck,
        actions_taken: list[str] | None = None,
    ) -> None:
        """Record a convergence check.

        Args:
            iteration: Iteration number
            check_result: Check result
            actions_taken: Actions taken
        """
        entry = ConvergenceHistoryEntry(
            iteration=iteration,
            check_result=check_result,
            actions_taken=actions_taken or [],
        )
        self._history.append(entry)

    def get_trend(self, metric: str) -> list[float]:
        """Get trend data for a metric.

        Args:
            metric: Metric name (call_graph_match_rate, documentation_similarity)

        Returns:
            List of metric values over iterations
        """
        values = []
        for entry in self._history:
            if entry.check_result:
                if metric == "call_graph_match_rate":
                    values.append(entry.check_result.call_graph_match_rate)
                elif metric == "documentation_similarity":
                    values.append(entry.check_result.documentation_similarity)
        return values

    def is_improving(self) -> bool:
        """Check if convergence is improving.

        Returns:
            True if metrics are trending upward
        """
        if len(self._history) < 2:
            return True  # Not enough data

        # Check call graph trend
        cg_trend = self.get_trend("call_graph_match_rate")
        if len(cg_trend) >= 2 and cg_trend[-1] < cg_trend[-2]:
            return False

        # Check documentation trend
        doc_trend = self.get_trend("documentation_similarity")
        if len(doc_trend) >= 2 and doc_trend[-1] < doc_trend[-2]:
            return False

        return True

    def is_stalled(self, window: int = 3) -> bool:
        """Check if convergence has stalled.

        Args:
            window: Number of iterations to check

        Returns:
            True if no progress in window
        """
        if len(self._history) < window:
            return False

        recent = self._history[-window:]

        # Check if all checks have same blocking count
        blocking_counts = [e.check_result.blocking_count for e in recent if e.check_result]

        if len(set(blocking_counts)) == 1 and blocking_counts[0] > 0:
            return True

        return False

    def get_recommendations(self) -> list[str]:
        """Get recommendations for improving convergence.

        Returns:
            List of recommendations
        """
        recommendations = []

        if not self._history:
            return ["Start documentation process"]

        latest = self._history[-1]
        if not latest.check_result:
            return recommendations

        # Get required actions
        actions = self._checker.get_required_actions(latest.check_result)
        recommendations.extend(actions)

        # Check for stall
        if self.is_stalled():
            recommendations.append(
                "Convergence appears stalled - consider manual intervention or adjusting criteria"
            )

        # Check for regression
        if not self.is_improving():
            recommendations.append("Metrics are declining - review recent changes and corrections")

        return recommendations

    def generate_report(self) -> dict[str, Any]:
        """Generate a convergence report.

        Returns:
            Report dictionary
        """
        final_status = "not_started"
        if self._history:
            latest = self._history[-1]
            if latest.check_result:
                final_status = latest.check_result.status.value

        return {
            "total_iterations": len(self._history),
            "final_status": final_status,
            "converged": (
                self._history[-1].check_result.converged
                if self._history and self._history[-1].check_result
                else False
            ),
            "history": [entry.to_dict() for entry in self._history],
            "trends": {
                "call_graph_match_rate": self.get_trend("call_graph_match_rate"),
                "documentation_similarity": self.get_trend("documentation_similarity"),
            },
            "is_improving": self.is_improving(),
            "is_stalled": self.is_stalled(),
            "recommendations": self.get_recommendations(),
        }


def calculate_similarity(
    text_a: str,
    text_b: str,
    method: str = "jaccard",
) -> float:
    """Calculate similarity between two texts.

    Args:
        text_a: First text
        text_b: Second text
        method: Similarity method (jaccard, cosine)

    Returns:
        Similarity score 0.0 to 1.0
    """
    if text_a == text_b:
        return 1.0

    if not text_a or not text_b:
        return 0.0

    if method == "jaccard":
        # Jaccard similarity on words
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        intersection = words_a & words_b
        union = words_a | words_b

        if not union:
            return 0.0

        return len(intersection) / len(union)

    elif method == "cosine":
        # Simple cosine similarity on word counts
        from collections import Counter

        words_a = Counter(text_a.lower().split())
        words_b = Counter(text_b.lower().split())

        all_words = set(words_a.keys()) | set(words_b.keys())

        dot_product = sum(words_a.get(w, 0) * words_b.get(w, 0) for w in all_words)
        magnitude_a = sum(v**2 for v in words_a.values()) ** 0.5
        magnitude_b = sum(v**2 for v in words_b.values()) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    return 0.0
