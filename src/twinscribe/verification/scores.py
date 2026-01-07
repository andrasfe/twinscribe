"""
CrossCheck Verification Framework - Score Aggregation System.

This module provides comprehensive scoring and quality grading for
verification results. It implements the VerificationScores dataclass
from the specification with weighted aggregation and quality assessment.

Key Components:
    - VerificationScores: Aggregated scores from all strategies
    - VerificationThresholds: Configurable pass/fail thresholds
    - QualityGrade: Letter grades (A/B/C/F) for documentation quality
    - ScoreAnalyzer: Utilities for analyzing and comparing scores
    - QualityReport: Detailed quality assessment report

The scoring system uses weighted aggregation:
    - Q&A: 15% (knowledge completeness)
    - Reconstruction: 20% (implementation detail)
    - Scenario: 20% (behavioral accuracy)
    - Mutation: 15% (boundary precision)
    - Impact: 15% (dependency tracking)
    - Test: 15% (behavioral correctness)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class QualityGrade(str, Enum):
    """Quality grade for documentation.

    Grades are assigned based on overall verification score:
        A: >= 0.95 - Production ready, comprehensive documentation
        B: >= 0.85 - Minor gaps, generally usable
        C: >= 0.70 - Significant gaps, needs improvement
        F: < 0.70 - Major revision needed, not production ready
    """

    A = "A"  # Production ready
    B = "B"  # Minor gaps
    C = "C"  # Significant gaps
    F = "F"  # Major revision needed


class WeaknessArea(str, Enum):
    """Areas that can be identified as weaknesses."""

    QA_KNOWLEDGE = "Q&A Knowledge"
    IMPLEMENTATION_DETAILS = "Implementation Details"
    EXECUTION_BEHAVIOR = "Execution Behavior"
    BOUNDARY_PRECISION = "Boundary Precision"
    DEPENDENCY_TRACKING = "Dependency Tracking"
    BEHAVIORAL_ACCURACY = "Behavioral Accuracy"


class VerificationThresholds(BaseModel):
    """Configurable thresholds for verification pass/fail.

    Attributes:
        min_overall_quality: Minimum overall score to pass
        min_qa_score: Minimum Q&A score
        min_reconstruction_score: Minimum reconstruction score
        min_scenario_score: Minimum scenario score
        min_mutation_score: Minimum mutation score
        min_impact_score: Minimum impact score
        min_test_pass_rate: Minimum test pass rate
        max_adversarial_findings: Maximum allowed adversarial findings
    """

    min_overall_quality: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum overall score to pass",
    )
    min_qa_score: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum Q&A score",
    )
    min_reconstruction_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum reconstruction score",
    )
    min_scenario_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum scenario score",
    )
    min_mutation_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum mutation score",
    )
    min_impact_score: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum impact score",
    )
    min_test_pass_rate: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum test pass rate",
    )
    max_adversarial_findings: int = Field(
        default=5,
        ge=0,
        description="Maximum allowed adversarial findings",
    )


class VerificationScores(BaseModel):
    """Aggregated scores from all verification strategies.

    This is the primary score container implementing the specification's
    VerificationScores dataclass with weighted aggregation and quality
    grading capabilities.

    Attributes:
        qa_score: Percentage of questions answered correctly (0.0-1.0)
        reconstruction_score: Percentage of masked elements reconstructed (0.0-1.0)
        scenario_score: Percentage of execution traces correct (0.0-1.0)
        mutation_score: Percentage of mutations detectable from docs (0.0-1.0)
        impact_score: Percentage of impacts correctly predicted (0.0-1.0)
        adversarial_findings: Number of issues found by cross-review
        test_pass_rate: Percentage of generated tests that pass (0.0-1.0)

    The overall quality score is computed using these weights:
        - Q&A: 15%
        - Reconstruction: 20%
        - Scenario: 20%
        - Mutation: 15%
        - Impact: 15%
        - Test: 15%
    """

    qa_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Q&A interrogation score",
    )
    reconstruction_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Masked reconstruction score",
    )
    scenario_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Scenario walkthrough score",
    )
    mutation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mutation detection score",
    )
    impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Impact analysis score",
    )
    adversarial_findings: int = Field(
        default=0,
        ge=0,
        description="Adversarial review finding count",
    )
    test_pass_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Test generation pass rate",
    )

    # Weights for overall quality calculation
    _WEIGHTS = {
        "qa": 0.15,
        "reconstruction": 0.20,
        "scenario": 0.20,
        "mutation": 0.15,
        "impact": 0.15,
        "test": 0.15,
    }

    @computed_field
    @property
    def overall_quality(self) -> float:
        """Weighted overall documentation quality score.

        Combines all strategy scores using predefined weights:
        - Q&A: 15% (knowledge completeness)
        - Reconstruction: 20% (implementation detail)
        - Scenario: 20% (behavioral accuracy)
        - Mutation: 15% (boundary precision)
        - Impact: 15% (dependency tracking)
        - Test: 15% (behavioral correctness)

        Returns:
            Float between 0.0 and 1.0
        """
        return (
            self.qa_score * self._WEIGHTS["qa"]
            + self.reconstruction_score * self._WEIGHTS["reconstruction"]
            + self.scenario_score * self._WEIGHTS["scenario"]
            + self.mutation_score * self._WEIGHTS["mutation"]
            + self.impact_score * self._WEIGHTS["impact"]
            + self.test_pass_rate * self._WEIGHTS["test"]
        )

    @computed_field
    @property
    def quality_grade(self) -> QualityGrade:
        """Assign a letter grade based on overall quality.

        Grading scale:
        - A: >= 0.95 (Production ready)
        - B: >= 0.85 (Minor gaps)
        - C: >= 0.70 (Significant gaps)
        - F: < 0.70 (Major revision needed)

        Returns:
            QualityGrade enum value
        """
        score = self.overall_quality
        if score >= 0.95:
            return QualityGrade.A
        elif score >= 0.85:
            return QualityGrade.B
        elif score >= 0.70:
            return QualityGrade.C
        else:
            return QualityGrade.F

    @computed_field
    @property
    def is_passing(self) -> bool:
        """Whether the scores meet minimum passing criteria.

        Default passing threshold is overall_quality >= 0.85 (grade B or better).
        """
        return self.quality_grade in {QualityGrade.A, QualityGrade.B}

    def get_weakest_areas(self, count: int = 3) -> list[str]:
        """Identify verification areas needing most improvement.

        Args:
            count: Number of weakest areas to return

        Returns:
            List of area names sorted by score (ascending)
        """
        scores = {
            WeaknessArea.QA_KNOWLEDGE.value: self.qa_score,
            WeaknessArea.IMPLEMENTATION_DETAILS.value: self.reconstruction_score,
            WeaknessArea.EXECUTION_BEHAVIOR.value: self.scenario_score,
            WeaknessArea.BOUNDARY_PRECISION.value: self.mutation_score,
            WeaknessArea.DEPENDENCY_TRACKING.value: self.impact_score,
            WeaknessArea.BEHAVIORAL_ACCURACY.value: self.test_pass_rate,
        }

        return sorted(scores.keys(), key=lambda k: scores[k])[:count]

    def get_strongest_areas(self, count: int = 3) -> list[str]:
        """Identify verification areas performing best.

        Args:
            count: Number of strongest areas to return

        Returns:
            List of area names sorted by score (descending)
        """
        scores = {
            WeaknessArea.QA_KNOWLEDGE.value: self.qa_score,
            WeaknessArea.IMPLEMENTATION_DETAILS.value: self.reconstruction_score,
            WeaknessArea.EXECUTION_BEHAVIOR.value: self.scenario_score,
            WeaknessArea.BOUNDARY_PRECISION.value: self.mutation_score,
            WeaknessArea.DEPENDENCY_TRACKING.value: self.impact_score,
            WeaknessArea.BEHAVIORAL_ACCURACY.value: self.test_pass_rate,
        }

        return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:count]

    def get_failing_thresholds(self, thresholds: VerificationThresholds) -> list[str]:
        """Get list of thresholds that are not met.

        Args:
            thresholds: Thresholds to check against

        Returns:
            List of threshold names that failed
        """
        failures = []

        if self.overall_quality < thresholds.min_overall_quality:
            failures.append("overall_quality")
        if self.qa_score < thresholds.min_qa_score:
            failures.append("qa_score")
        if self.reconstruction_score < thresholds.min_reconstruction_score:
            failures.append("reconstruction_score")
        if self.scenario_score < thresholds.min_scenario_score:
            failures.append("scenario_score")
        if self.mutation_score < thresholds.min_mutation_score:
            failures.append("mutation_score")
        if self.impact_score < thresholds.min_impact_score:
            failures.append("impact_score")
        if self.test_pass_rate < thresholds.min_test_pass_rate:
            failures.append("test_pass_rate")
        if self.adversarial_findings > thresholds.max_adversarial_findings:
            failures.append("adversarial_findings")

        return failures

    def meets_thresholds(self, thresholds: VerificationThresholds) -> bool:
        """Check if all thresholds are met.

        Args:
            thresholds: Thresholds to check against

        Returns:
            True if all thresholds are met
        """
        return len(self.get_failing_thresholds(thresholds)) == 0

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary for reporting.

        Returns:
            Dictionary with all scores and computed values
        """
        return {
            "scores": {
                "qa_score": self.qa_score,
                "reconstruction_score": self.reconstruction_score,
                "scenario_score": self.scenario_score,
                "mutation_score": self.mutation_score,
                "impact_score": self.impact_score,
                "adversarial_findings": self.adversarial_findings,
                "test_pass_rate": self.test_pass_rate,
            },
            "overall_quality": self.overall_quality,
            "quality_grade": self.quality_grade.value,
            "is_passing": self.is_passing,
            "weakest_areas": self.get_weakest_areas(),
            "strongest_areas": self.get_strongest_areas(),
        }


@dataclass
class ScoreComparison:
    """Comparison between two sets of verification scores.

    Useful for tracking improvement over iterations.

    Attributes:
        before: Previous scores
        after: Current scores
        improvements: Areas that improved
        regressions: Areas that regressed
        unchanged: Areas with no significant change
    """

    before: VerificationScores
    after: VerificationScores
    improvements: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def overall_delta(self) -> float:
        """Change in overall quality score."""
        return self.after.overall_quality - self.before.overall_quality

    @property
    def grade_improved(self) -> bool:
        """Whether the quality grade improved."""
        grade_order = [QualityGrade.F, QualityGrade.C, QualityGrade.B, QualityGrade.A]
        before_idx = grade_order.index(self.before.quality_grade)
        after_idx = grade_order.index(self.after.quality_grade)
        return after_idx > before_idx

    @property
    def is_improvement(self) -> bool:
        """Whether there was overall improvement."""
        return self.overall_delta > 0.01  # 1% threshold


class ScoreAnalyzer:
    """Utilities for analyzing verification scores.

    Provides methods for:
    - Comparing scores across iterations
    - Identifying improvement opportunities
    - Generating improvement recommendations
    - Computing trends across multiple runs
    """

    CHANGE_THRESHOLD = 0.05  # 5% change threshold

    @classmethod
    def compare(
        cls,
        before: VerificationScores,
        after: VerificationScores,
    ) -> ScoreComparison:
        """Compare two sets of scores.

        Args:
            before: Previous scores
            after: Current scores

        Returns:
            ScoreComparison with categorized changes
        """
        comparisons = {
            "Q&A": (before.qa_score, after.qa_score),
            "Reconstruction": (before.reconstruction_score, after.reconstruction_score),
            "Scenario": (before.scenario_score, after.scenario_score),
            "Mutation": (before.mutation_score, after.mutation_score),
            "Impact": (before.impact_score, after.impact_score),
            "Test": (before.test_pass_rate, after.test_pass_rate),
        }

        improvements = []
        regressions = []
        unchanged = []

        for area, (b, a) in comparisons.items():
            delta = a - b
            if delta > cls.CHANGE_THRESHOLD:
                improvements.append(area)
            elif delta < -cls.CHANGE_THRESHOLD:
                regressions.append(area)
            else:
                unchanged.append(area)

        return ScoreComparison(
            before=before,
            after=after,
            improvements=improvements,
            regressions=regressions,
            unchanged=unchanged,
        )

    @classmethod
    def get_improvement_recommendations(
        cls,
        scores: VerificationScores,
        thresholds: VerificationThresholds | None = None,
    ) -> list[dict[str, str]]:
        """Generate recommendations for improving scores.

        Args:
            scores: Current verification scores
            thresholds: Optional thresholds to check

        Returns:
            List of recommendations with area, priority, and action
        """
        thresholds = thresholds or VerificationThresholds()
        recommendations = []

        # Check each area
        areas = [
            (
                "qa_score",
                scores.qa_score,
                thresholds.min_qa_score,
                "Q&A Knowledge",
                "Add more complete documentation for return values, exceptions, and edge cases",
            ),
            (
                "reconstruction_score",
                scores.reconstruction_score,
                thresholds.min_reconstruction_score,
                "Implementation Details",
                "Document specific values, thresholds, and business rules more precisely",
            ),
            (
                "scenario_score",
                scores.scenario_score,
                thresholds.min_scenario_score,
                "Execution Behavior",
                "Improve call graph documentation and side effect descriptions",
            ),
            (
                "mutation_score",
                scores.mutation_score,
                thresholds.min_mutation_score,
                "Boundary Precision",
                "Document boundary conditions and comparison operators more precisely",
            ),
            (
                "impact_score",
                scores.impact_score,
                thresholds.min_impact_score,
                "Dependency Tracking",
                "Improve documentation of callers, callees, and dependency relationships",
            ),
            (
                "test_pass_rate",
                scores.test_pass_rate,
                thresholds.min_test_pass_rate,
                "Behavioral Accuracy",
                "Ensure documentation accurately reflects actual code behavior",
            ),
        ]

        for _field, score, threshold, area, action in areas:
            if score < threshold:
                gap = threshold - score
                priority = "critical" if gap > 0.2 else ("high" if gap > 0.1 else "medium")
                recommendations.append(
                    {
                        "area": area,
                        "current_score": f"{score:.0%}",
                        "target_score": f"{threshold:.0%}",
                        "gap": f"{gap:.0%}",
                        "priority": priority,
                        "action": action,
                    }
                )

        # Sort by gap (descending)
        recommendations.sort(key=lambda r: float(r["gap"].rstrip("%")), reverse=True)
        return recommendations

    @classmethod
    def compute_trend(
        cls,
        score_history: list[VerificationScores],
    ) -> dict[str, Any]:
        """Compute trend across multiple score samples.

        Args:
            score_history: List of scores in chronological order

        Returns:
            Dictionary with trend analysis
        """
        if len(score_history) < 2:
            return {
                "trend": "insufficient_data",
                "samples": len(score_history),
            }

        # Calculate deltas between consecutive samples
        deltas = []
        for i in range(1, len(score_history)):
            delta = score_history[i].overall_quality - score_history[i - 1].overall_quality
            deltas.append(delta)

        avg_delta = sum(deltas) / len(deltas)
        first = score_history[0].overall_quality
        last = score_history[-1].overall_quality
        total_change = last - first

        # Determine trend
        if avg_delta > 0.02:
            trend = "improving"
        elif avg_delta < -0.02:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "samples": len(score_history),
            "average_delta": avg_delta,
            "total_change": total_change,
            "first_score": first,
            "last_score": last,
            "first_grade": score_history[0].quality_grade.value,
            "last_grade": score_history[-1].quality_grade.value,
        }


@dataclass
class QualityReport:
    """Detailed quality assessment report.

    Provides comprehensive analysis of verification results including
    scores, grades, recommendations, and trends.

    Attributes:
        component_id: Component being assessed
        scores: Verification scores
        thresholds: Thresholds used for assessment
        timestamp: When report was generated
        recommendations: Improvement recommendations
        comparison: Optional comparison to previous scores
    """

    component_id: str
    scores: VerificationScores
    thresholds: VerificationThresholds = field(default_factory=VerificationThresholds)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommendations: list[dict[str, str]] = field(default_factory=list)
    comparison: ScoreComparison | None = None

    def __post_init__(self):
        """Generate recommendations after initialization."""
        if not self.recommendations:
            self.recommendations = ScoreAnalyzer.get_improvement_recommendations(
                self.scores, self.thresholds
            )

    @property
    def passed(self) -> bool:
        """Whether the component passed verification."""
        return self.scores.meets_thresholds(self.thresholds)

    @property
    def grade(self) -> QualityGrade:
        """Quality grade."""
        return self.scores.quality_grade

    @property
    def failing_areas(self) -> list[str]:
        """Areas failing to meet thresholds."""
        return self.scores.get_failing_thresholds(self.thresholds)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "grade": self.grade.value,
            "overall_quality": self.scores.overall_quality,
            "scores": self.scores.to_summary_dict(),
            "failing_areas": self.failing_areas,
            "recommendations": self.recommendations,
        }

        if self.comparison:
            result["comparison"] = {
                "overall_delta": self.comparison.overall_delta,
                "grade_improved": self.comparison.grade_improved,
                "improvements": self.comparison.improvements,
                "regressions": self.comparison.regressions,
            }

        return result

    def to_markdown(self) -> str:
        """Generate markdown report.

        Returns:
            Markdown formatted report string
        """
        lines = [
            f"# Quality Report: {self.component_id}",
            "",
            f"**Generated:** {self.timestamp.isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Grade:** {self.grade.value}",
            f"- **Overall Quality:** {self.scores.overall_quality:.1%}",
            f"- **Status:** {'PASSED' if self.passed else 'FAILED'}",
            "",
            "## Scores",
            "",
            "| Area | Score | Threshold | Status |",
            "|------|-------|-----------|--------|",
        ]

        score_data = [
            ("Q&A", self.scores.qa_score, self.thresholds.min_qa_score),
            (
                "Reconstruction",
                self.scores.reconstruction_score,
                self.thresholds.min_reconstruction_score,
            ),
            ("Scenario", self.scores.scenario_score, self.thresholds.min_scenario_score),
            ("Mutation", self.scores.mutation_score, self.thresholds.min_mutation_score),
            ("Impact", self.scores.impact_score, self.thresholds.min_impact_score),
            ("Test Pass Rate", self.scores.test_pass_rate, self.thresholds.min_test_pass_rate),
        ]

        for area, score, threshold in score_data:
            status = "PASS" if score >= threshold else "FAIL"
            lines.append(f"| {area} | {score:.1%} | {threshold:.1%} | {status} |")

        lines.extend(
            [
                "",
                f"**Adversarial Findings:** {self.scores.adversarial_findings}",
                "",
            ]
        )

        if self.recommendations:
            lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )
            for i, rec in enumerate(self.recommendations, 1):
                lines.extend(
                    [
                        f"### {i}. {rec['area']} ({rec['priority'].upper()})",
                        "",
                        f"- Current: {rec['current_score']}",
                        f"- Target: {rec['target_score']}",
                        f"- Gap: {rec['gap']}",
                        f"- Action: {rec['action']}",
                        "",
                    ]
                )

        if self.comparison:
            lines.extend(
                [
                    "## Comparison to Previous",
                    "",
                    f"- **Overall Change:** {self.comparison.overall_delta:+.1%}",
                    f"- **Grade Improved:** {'Yes' if self.comparison.grade_improved else 'No'}",
                ]
            )
            if self.comparison.improvements:
                lines.append(f"- **Improved:** {', '.join(self.comparison.improvements)}")
            if self.comparison.regressions:
                lines.append(f"- **Regressed:** {', '.join(self.comparison.regressions)}")

        return "\n".join(lines)


class ScoreAggregator:
    """Aggregates scores across multiple components.

    Provides methods for computing aggregate statistics across
    a codebase or subset of components.
    """

    @classmethod
    def aggregate(
        cls,
        component_scores: dict[str, VerificationScores],
    ) -> VerificationScores:
        """Aggregate scores across components.

        Args:
            component_scores: Mapping of component ID to scores

        Returns:
            Aggregated VerificationScores
        """
        if not component_scores:
            return VerificationScores()

        scores = list(component_scores.values())
        n = len(scores)

        return VerificationScores(
            qa_score=sum(s.qa_score for s in scores) / n,
            reconstruction_score=sum(s.reconstruction_score for s in scores) / n,
            scenario_score=sum(s.scenario_score for s in scores) / n,
            mutation_score=sum(s.mutation_score for s in scores) / n,
            impact_score=sum(s.impact_score for s in scores) / n,
            adversarial_findings=sum(s.adversarial_findings for s in scores),
            test_pass_rate=sum(s.test_pass_rate for s in scores) / n,
        )

    @classmethod
    def get_statistics(
        cls,
        component_scores: dict[str, VerificationScores],
    ) -> dict[str, Any]:
        """Compute statistics across components.

        Args:
            component_scores: Mapping of component ID to scores

        Returns:
            Dictionary with statistical analysis
        """
        if not component_scores:
            return {"count": 0}

        scores = list(component_scores.values())
        overall_scores = [s.overall_quality for s in scores]
        grades = [s.quality_grade for s in scores]

        # Grade distribution
        grade_dist = {
            QualityGrade.A: sum(1 for g in grades if g == QualityGrade.A),
            QualityGrade.B: sum(1 for g in grades if g == QualityGrade.B),
            QualityGrade.C: sum(1 for g in grades if g == QualityGrade.C),
            QualityGrade.F: sum(1 for g in grades if g == QualityGrade.F),
        }

        return {
            "count": len(scores),
            "mean_quality": sum(overall_scores) / len(overall_scores),
            "min_quality": min(overall_scores),
            "max_quality": max(overall_scores),
            "median_quality": sorted(overall_scores)[len(overall_scores) // 2],
            "grade_distribution": {k.value: v for k, v in grade_dist.items()},
            "passing_rate": sum(1 for s in scores if s.is_passing) / len(scores),
            "aggregate_scores": cls.aggregate(component_scores).to_summary_dict(),
        }

    @classmethod
    def get_worst_components(
        cls,
        component_scores: dict[str, VerificationScores],
        count: int = 10,
    ) -> list[tuple[str, VerificationScores]]:
        """Get components with worst verification scores.

        Args:
            component_scores: Mapping of component ID to scores
            count: Number of worst components to return

        Returns:
            List of (component_id, scores) tuples sorted by quality (ascending)
        """
        sorted_items = sorted(
            component_scores.items(),
            key=lambda x: x[1].overall_quality,
        )
        return sorted_items[:count]

    @classmethod
    def get_best_components(
        cls,
        component_scores: dict[str, VerificationScores],
        count: int = 10,
    ) -> list[tuple[str, VerificationScores]]:
        """Get components with best verification scores.

        Args:
            component_scores: Mapping of component ID to scores
            count: Number of best components to return

        Returns:
            List of (component_id, scores) tuples sorted by quality (descending)
        """
        sorted_items = sorted(
            component_scores.items(),
            key=lambda x: x[1].overall_quality,
            reverse=True,
        )
        return sorted_items[:count]
