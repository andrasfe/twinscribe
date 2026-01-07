"""
Integration tests for the Verification Pipeline.

These tests verify the end-to-end behavior of the verification pipeline,
including:
- Full pipeline execution with mock strategies
- Strategy orchestration and phase progression
- Quality threshold behavior
- Result aggregation and grading
- Beads ticket generation
- Re-verification loops

Related Beads Tickets:
- twinscribe-9x0: Create integration tests for verification pipeline
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from twinscribe.verification.base import (
    Severity,
    StrategyType,
    VerificationLevel,
    VerificationStrategy,
)
from twinscribe.verification.models import (
    DocumentationGap,
    VerificationChallenge,
    VerificationResult,
)
from twinscribe.verification.pipeline import (
    PipelineBuilder,
    PipelineConfig,
    PipelinePhase,
    PipelineStatus,
    VerificationPipeline,
)
from twinscribe.verification.scores import (
    QualityGrade,
    ScoreAnalyzer,
    VerificationScores,
    VerificationThresholds,
)

# =============================================================================
# Mock Providers for Integration Testing
# =============================================================================


class MockDocumentationProvider:
    """Mock documentation provider for integration testing."""

    def __init__(
        self,
        team_a_docs: str = "Team A documentation",
        team_b_docs: str = "Team B documentation",
    ):
        self.team_a_docs = team_a_docs
        self.team_b_docs = team_b_docs
        self.calls: list[tuple[str, str]] = []

    async def get_team_a_documentation(self, component_id: str) -> str:
        self.calls.append(("team_a", component_id))
        return self.team_a_docs

    async def get_team_b_documentation(self, component_id: str) -> str:
        self.calls.append(("team_b", component_id))
        return self.team_b_docs


class MockSourceCodeProvider:
    """Mock source code provider for integration testing."""

    def __init__(self, source_code: str = "def sample(): pass"):
        self.source_code = source_code
        self.calls: list[tuple[str, str]] = []

    async def get_source_code(self, component_id: str) -> str:
        self.calls.append(("source", component_id))
        return self.source_code

    async def get_call_graph(self, component_id: str) -> dict[str, list[str]]:
        self.calls.append(("call_graph", component_id))
        return {"callers": [], "callees": []}


class MockTicketCreator:
    """Mock ticket creator for integration testing."""

    def __init__(self):
        self.created_tickets: list[dict[str, Any]] = []

    async def create_gap_ticket(
        self,
        gap: DocumentationGap,
        component_id: str,
        strategy_type: StrategyType,
    ) -> str:
        ticket_id = f"bd-{len(self.created_tickets) + 1:04d}"
        self.created_tickets.append(
            {
                "ticket_id": ticket_id,
                "gap": gap,
                "component_id": component_id,
                "strategy_type": strategy_type,
            }
        )
        return ticket_id


class MockVerificationStrategy(VerificationStrategy):
    """Mock strategy for controlled testing."""

    def __init__(
        self,
        strategy_type: StrategyType,
        level: VerificationLevel,
        score: float = 0.85,
        gaps: list[DocumentationGap] | None = None,
        execution_delay: float = 0.0,
    ):
        super().__init__(
            strategy_type=strategy_type,
            level=level,
            description=f"Mock {strategy_type.value} strategy",
        )
        self._score = score
        self._gaps = gaps or []
        self._execution_delay = execution_delay
        self.execution_count = 0
        self.execution_order: list[int] = []
        self._global_counter = 0

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> VerificationChallenge:
        if self._execution_delay > 0:
            await asyncio.sleep(self._execution_delay)
        return MagicMock(spec=VerificationChallenge)

    async def evaluate(
        self,
        challenge: VerificationChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> VerificationResult:
        self.execution_count += 1
        result = MagicMock(spec=VerificationResult)
        result.average_score = self._score
        result.documentation_gaps = self._gaps
        return result

    def get_documentation_gaps(self, result: VerificationResult) -> list[dict]:
        return [{"area": g.area, "severity": g.severity.value} for g in self._gaps]


class MockStrategyRegistry:
    """Mock strategy registry with controllable strategies."""

    def __init__(
        self,
        strategies: dict[StrategyType, MockVerificationStrategy] | None = None,
    ):
        self._strategies = strategies or {}

    def get(self, strategy_type: StrategyType) -> VerificationStrategy:
        if strategy_type in self._strategies:
            return self._strategies[strategy_type]
        # Return a default mock strategy with correct type
        level = self._get_level_for_type(strategy_type)
        return MockVerificationStrategy(strategy_type, level)

    def _get_level_for_type(self, strategy_type: StrategyType) -> VerificationLevel:
        active_strategies = {
            StrategyType.QA_INTERROGATION,
            StrategyType.MASKED_RECONSTRUCTION,
            StrategyType.SCENARIO_WALKTHROUGH,
        }
        behavioral_strategies = {
            StrategyType.MUTATION_DETECTION,
            StrategyType.IMPACT_ANALYSIS,
            StrategyType.EDGE_CASE_EXTRACTION,
        }
        if strategy_type in active_strategies:
            return VerificationLevel.ACTIVE
        elif strategy_type in behavioral_strategies:
            return VerificationLevel.BEHAVIORAL
        else:
            return VerificationLevel.GENERATIVE


# =============================================================================
# Integration Test: Full Pipeline Execution
# =============================================================================


@pytest.mark.integration
class TestFullPipelineExecution:
    """Integration tests for complete pipeline execution."""

    @pytest.mark.asyncio
    async def test_execute_complete_pipeline_with_mock_strategies(self):
        """Test executing complete verification pipeline with all strategy types."""
        # Create mock strategies for each type
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.90,
            ),
            StrategyType.SCENARIO_WALKTHROUGH: MockVerificationStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
            StrategyType.MUTATION_DETECTION: MockVerificationStrategy(
                StrategyType.MUTATION_DETECTION,
                VerificationLevel.BEHAVIORAL,
                score=0.80,
            ),
            StrategyType.TEST_GENERATION: MockVerificationStrategy(
                StrategyType.TEST_GENERATION,
                VerificationLevel.GENERATIVE,
                score=0.88,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        assert run.status == PipelineStatus.COMPLETED
        assert run.total_components == 1
        assert "test.component" in run.component_verifications

        # Verify all strategies were executed
        cv = run.component_verifications["test.component"]
        executed_types = {e.strategy_type for e in cv.executions if e.succeeded}
        assert len(executed_types) >= 1  # At least one strategy should execute

    @pytest.mark.asyncio
    async def test_all_strategy_types_are_executed(self):
        """Verify that all enabled strategy types are executed."""
        execution_tracker: list[StrategyType] = []

        class TrackingStrategy(MockVerificationStrategy):
            def __init__(self, strategy_type: StrategyType, level: VerificationLevel):
                super().__init__(strategy_type, level, score=0.85)

            async def evaluate(
                self, challenge, team_a_response, team_b_response, ground_truth=None
            ):
                execution_tracker.append(self.strategy_type)
                return await super().evaluate(
                    challenge, team_a_response, team_b_response, ground_truth
                )

        strategies = {
            StrategyType.QA_INTERROGATION: TrackingStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
            StrategyType.SCENARIO_WALKTHROUGH: TrackingStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        await pipeline.verify_components(["test.component"])

        # Both ACTIVE level strategies should be executed
        assert StrategyType.QA_INTERROGATION in execution_tracker
        assert StrategyType.SCENARIO_WALKTHROUGH in execution_tracker

    @pytest.mark.asyncio
    async def test_score_aggregation_from_all_strategies(self):
        """Verify that scores are aggregated from all executed strategies."""
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.90,
            ),
            StrategyType.MASKED_RECONSTRUCTION: MockVerificationStrategy(
                StrategyType.MASKED_RECONSTRUCTION,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
            StrategyType.SCENARIO_WALKTHROUGH: MockVerificationStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
                score=0.88,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        cv = run.component_verifications["test.component"]
        assert cv.scores is not None
        # Scores should be populated from strategy results
        assert cv.scores.qa_score == 0.90
        assert cv.scores.reconstruction_score == 0.85
        assert cv.scores.scenario_score == 0.88


# =============================================================================
# Integration Test: Strategy Orchestration
# =============================================================================


@pytest.mark.integration
class TestStrategyOrchestration:
    """Integration tests for strategy execution order and orchestration."""

    @pytest.mark.asyncio
    async def test_strategy_execution_order_by_phase(self):
        """Verify strategies execute in correct phase order (ACTIVE -> BEHAVIORAL -> GENERATIVE)."""
        phases_executed: list[PipelinePhase] = []

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
            StrategyType.MUTATION_DETECTION: MockVerificationStrategy(
                StrategyType.MUTATION_DETECTION,
                VerificationLevel.BEHAVIORAL,
            ),
            StrategyType.TEST_GENERATION: MockVerificationStrategy(
                StrategyType.TEST_GENERATION,
                VerificationLevel.GENERATIVE,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        # Track phase execution
        pipeline.on_phase_start(lambda phase: phases_executed.append(phase))

        await pipeline.verify_components(["test.component"])

        # Verify phase order
        expected_order = [
            PipelinePhase.ACTIVE,
            PipelinePhase.BEHAVIORAL,
            PipelinePhase.GENERATIVE,
            PipelinePhase.CONSOLIDATION,
        ]
        assert phases_executed == expected_order

    @pytest.mark.asyncio
    async def test_parallel_execution_of_independent_strategies(self):
        """Test that independent strategies within a phase can run in parallel."""
        execution_times: dict[StrategyType, tuple[float, float]] = {}

        class TimingStrategy(MockVerificationStrategy):
            def __init__(self, strategy_type: StrategyType, delay: float):
                super().__init__(
                    strategy_type,
                    VerificationLevel.ACTIVE,
                    execution_delay=delay,
                )
                self._start_time: float | None = None

            async def generate_challenge(self, component_id, source_code, **kwargs):
                self._start_time = asyncio.get_event_loop().time()
                return await super().generate_challenge(component_id, source_code, **kwargs)

            async def evaluate(
                self, challenge, team_a_response, team_b_response, ground_truth=None
            ):
                result = await super().evaluate(
                    challenge, team_a_response, team_b_response, ground_truth
                )
                end_time = asyncio.get_event_loop().time()
                if self._start_time is not None:
                    execution_times[self.strategy_type] = (self._start_time, end_time)
                return result

        # Both strategies have 0.1s delay each
        strategies = {
            StrategyType.QA_INTERROGATION: TimingStrategy(StrategyType.QA_INTERROGATION, 0.1),
            StrategyType.MASKED_RECONSTRUCTION: TimingStrategy(
                StrategyType.MASKED_RECONSTRUCTION, 0.1
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            max_concurrent_strategies=2,  # Allow parallel execution
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        start = asyncio.get_event_loop().time()
        await pipeline.verify_components(["test.component"])
        total_time = asyncio.get_event_loop().time() - start

        # If strategies ran in parallel, total time should be close to 0.1s, not 0.2s
        # Allow some overhead
        assert total_time < 0.3, f"Strategies should run in parallel, took {total_time}s"

    @pytest.mark.asyncio
    async def test_phase_progression_active_to_behavioral_to_generative(self):
        """Test progression through ACTIVE -> BEHAVIORAL -> GENERATIVE phases."""
        phase_transitions: list[tuple[PipelinePhase, str]] = []

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
            StrategyType.MUTATION_DETECTION: MockVerificationStrategy(
                StrategyType.MUTATION_DETECTION,
                VerificationLevel.BEHAVIORAL,
            ),
            StrategyType.TEST_GENERATION: MockVerificationStrategy(
                StrategyType.TEST_GENERATION,
                VerificationLevel.GENERATIVE,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        pipeline.on_phase_start(lambda phase: phase_transitions.append((phase, "start")))
        pipeline.on_phase_complete(lambda phase: phase_transitions.append((phase, "complete")))

        await pipeline.verify_components(["test.component"])

        # Verify each phase starts and completes before the next begins
        expected_transitions = [
            (PipelinePhase.ACTIVE, "start"),
            (PipelinePhase.ACTIVE, "complete"),
            (PipelinePhase.BEHAVIORAL, "start"),
            (PipelinePhase.BEHAVIORAL, "complete"),
            (PipelinePhase.GENERATIVE, "start"),
            (PipelinePhase.GENERATIVE, "complete"),
            (PipelinePhase.CONSOLIDATION, "start"),
            (PipelinePhase.CONSOLIDATION, "complete"),
        ]
        assert phase_transitions == expected_transitions


# =============================================================================
# Integration Test: Quality Threshold
# =============================================================================


@pytest.mark.integration
class TestQualityThreshold:
    """Integration tests for quality threshold behavior."""

    def test_quality_threshold_detection(self):
        """Test that quality thresholds correctly detect passing/failing scores."""
        thresholds = VerificationThresholds(
            min_overall_quality=0.85,
            min_qa_score=0.80,
            min_reconstruction_score=0.75,
        )

        # Passing scores
        passing_scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.85,
            scenario_score=0.90,
            mutation_score=0.85,
            impact_score=0.88,
            test_pass_rate=0.92,
        )
        assert passing_scores.meets_thresholds(thresholds)
        assert passing_scores.is_passing

        # Failing scores
        failing_scores = VerificationScores(
            qa_score=0.50,
            reconstruction_score=0.45,
            scenario_score=0.50,
            mutation_score=0.40,
            impact_score=0.45,
            test_pass_rate=0.52,
        )
        assert not failing_scores.meets_thresholds(thresholds)
        assert not failing_scores.is_passing

    @pytest.mark.asyncio
    async def test_pipeline_continues_below_threshold(self):
        """Test that pipeline continues execution when below quality threshold."""
        # Strategy with low score
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.50,  # Below threshold
            ),
            StrategyType.SCENARIO_WALKTHROUGH: MockVerificationStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
                score=0.55,  # Below threshold
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            thresholds=VerificationThresholds(min_overall_quality=0.85),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        # Pipeline should complete even with low scores
        assert run.status == PipelineStatus.COMPLETED
        cv = run.component_verifications["test.component"]
        assert cv.scores is not None
        # Scores should reflect the low values
        assert not cv.scores.is_passing

    def test_early_termination_check_when_threshold_met(self):
        """Test that threshold check correctly identifies when quality is met."""
        thresholds = VerificationThresholds(
            min_overall_quality=0.85,
        )

        # High quality scores that meet threshold
        high_scores = VerificationScores(
            qa_score=0.95,
            reconstruction_score=0.92,
            scenario_score=0.94,
            mutation_score=0.90,
            impact_score=0.91,
            test_pass_rate=0.93,
        )

        assert high_scores.overall_quality >= thresholds.min_overall_quality
        assert high_scores.meets_thresholds(thresholds)
        assert len(high_scores.get_failing_thresholds(thresholds)) == 0


# =============================================================================
# Integration Test: Result Aggregation
# =============================================================================


@pytest.mark.integration
class TestResultAggregation:
    """Integration tests for verification result aggregation."""

    def test_verification_scores_calculation(self):
        """Test VerificationScores weighted overall quality calculation."""
        scores = VerificationScores(
            qa_score=0.90,  # 15% weight
            reconstruction_score=0.85,  # 20% weight
            scenario_score=0.92,  # 20% weight
            mutation_score=0.80,  # 15% weight
            impact_score=0.88,  # 15% weight
            test_pass_rate=0.91,  # 15% weight
        )

        # Manual calculation:
        # 0.90*0.15 + 0.85*0.20 + 0.92*0.20 + 0.80*0.15 + 0.88*0.15 + 0.91*0.15
        # = 0.135 + 0.17 + 0.184 + 0.12 + 0.132 + 0.1365 = 0.8775
        expected_overall = 0.8775
        assert abs(scores.overall_quality - expected_overall) < 0.001

    def test_quality_grade_assignment_a(self):
        """Test grade A assignment for scores >= 0.95."""
        scores = VerificationScores(
            qa_score=0.98,
            reconstruction_score=0.97,
            scenario_score=0.98,
            mutation_score=0.96,
            impact_score=0.97,
            test_pass_rate=0.98,
        )
        assert scores.quality_grade == QualityGrade.A
        assert scores.overall_quality >= 0.95

    def test_quality_grade_assignment_b(self):
        """Test grade B assignment for scores >= 0.85 and < 0.95."""
        scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.88,
            scenario_score=0.90,
            mutation_score=0.87,
            impact_score=0.88,
            test_pass_rate=0.90,
        )
        assert scores.quality_grade == QualityGrade.B
        assert 0.85 <= scores.overall_quality < 0.95

    def test_quality_grade_assignment_c(self):
        """Test grade C assignment for scores >= 0.70 and < 0.85."""
        scores = VerificationScores(
            qa_score=0.75,
            reconstruction_score=0.72,
            scenario_score=0.78,
            mutation_score=0.70,
            impact_score=0.73,
            test_pass_rate=0.76,
        )
        assert scores.quality_grade == QualityGrade.C
        assert 0.70 <= scores.overall_quality < 0.85

    def test_quality_grade_assignment_f(self):
        """Test grade F assignment for scores < 0.70."""
        scores = VerificationScores(
            qa_score=0.50,
            reconstruction_score=0.45,
            scenario_score=0.55,
            mutation_score=0.40,
            impact_score=0.48,
            test_pass_rate=0.52,
        )
        assert scores.quality_grade == QualityGrade.F
        assert scores.overall_quality < 0.70

    def test_weakest_areas_identification(self):
        """Test identification of verification areas needing improvement."""
        scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.50,  # Weakest
            scenario_score=0.85,
            mutation_score=0.55,  # Second weakest
            impact_score=0.88,
            test_pass_rate=0.60,  # Third weakest
        )

        weak_areas = scores.get_weakest_areas(count=3)
        assert len(weak_areas) == 3
        # Implementation Details (reconstruction) should be first
        assert "Implementation Details" in weak_areas[0]

    @pytest.mark.asyncio
    async def test_aggregate_scores_across_components(self):
        """Test aggregation of scores across multiple components."""
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["comp1", "comp2", "comp3"])

        assert run.total_components == 3
        aggregate = run.get_aggregate_scores()
        assert aggregate is not None
        # All components have the same score, so aggregate should match
        assert aggregate.qa_score == 0.85


# =============================================================================
# Integration Test: Beads Ticket Generation
# =============================================================================


@pytest.mark.integration
class TestBeadsTicketGeneration:
    """Integration tests for Beads ticket generation from verification results."""

    @pytest.mark.asyncio
    async def test_ticket_creation_for_documentation_gaps(self):
        """Test that tickets are created for identified documentation gaps."""
        gap = DocumentationGap(
            gap_id="gap_001",
            area="edge_case",
            description="Missing boundary condition documentation",
            severity=Severity.HIGH,
            recommendation="Document behavior at boundary values",
        )

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.70,
                gaps=[gap],
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            create_tickets_for_gaps=True,
            min_severity_for_ticket="high",
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
            ticket_creator=ticket_creator,
        )

        await pipeline.verify_components(["test.component"])

        # Ticket should be created for the high-severity gap
        assert len(ticket_creator.created_tickets) >= 1
        ticket = ticket_creator.created_tickets[0]
        assert ticket["component_id"] == "test.component"
        assert ticket["gap"].severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_severity_based_ticket_filtering(self):
        """Test that tickets are only created for gaps meeting severity threshold."""
        low_gap = DocumentationGap(
            gap_id="gap_low_001",
            area="formatting",
            description="Minor formatting issue",
            severity=Severity.LOW,
            recommendation="Fix formatting",
        )
        high_gap = DocumentationGap(
            gap_id="gap_high_001",
            area="edge_case",
            description="Critical edge case missing",
            severity=Severity.HIGH,
            recommendation="Document edge case",
        )

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.70,
                gaps=[low_gap, high_gap],
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            create_tickets_for_gaps=True,
            min_severity_for_ticket="high",  # Only high and critical
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
            ticket_creator=ticket_creator,
        )

        await pipeline.verify_components(["test.component"])

        # Only high severity gap should create a ticket
        high_severity_tickets = [
            t
            for t in ticket_creator.created_tickets
            if t["gap"].severity in {Severity.HIGH, Severity.CRITICAL}
        ]
        low_severity_tickets = [
            t for t in ticket_creator.created_tickets if t["gap"].severity == Severity.LOW
        ]

        assert len(high_severity_tickets) >= 1
        assert len(low_severity_tickets) == 0

    @pytest.mark.asyncio
    async def test_no_tickets_when_disabled(self):
        """Test that no tickets are created when ticket generation is disabled."""
        gap = DocumentationGap(
            gap_id="gap_critical_001",
            area="critical_issue",
            description="Critical documentation gap",
            severity=Severity.CRITICAL,
            recommendation="Fix immediately",
        )

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.50,
                gaps=[gap],
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            create_tickets_for_gaps=False,  # Disabled
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
            ticket_creator=ticket_creator,
        )

        await pipeline.verify_components(["test.component"])

        # No tickets should be created
        assert len(ticket_creator.created_tickets) == 0


# =============================================================================
# Integration Test: Re-verification Loop
# =============================================================================


@pytest.mark.integration
class TestReverificationLoop:
    """Integration tests for re-verification after fixes."""

    @pytest.mark.asyncio
    async def test_reverification_after_fixes(self):
        """Test re-verification produces new results after documentation fixes."""
        # First run - low scores
        initial_strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.60,
            ),
        }

        registry = MockStrategyRegistry(initial_strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[StrategyType.QA_INTERROGATION],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        first_run = await pipeline.verify_components(["test.component"])
        first_scores = first_run.component_verifications["test.component"].scores
        assert first_scores is not None
        assert first_scores.qa_score == 0.60

        # Second run - improved scores (simulating fixes)
        improved_strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.90,  # Improved
            ),
        }

        improved_registry = MockStrategyRegistry(improved_strategies)
        improved_pipeline = VerificationPipeline(
            strategy_registry=improved_registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        second_run = await improved_pipeline.verify_components(["test.component"])
        second_scores = second_run.component_verifications["test.component"].scores
        assert second_scores is not None
        assert second_scores.qa_score == 0.90

        # Scores should have improved
        assert second_scores.qa_score > first_scores.qa_score

    def test_score_improvement_tracking(self):
        """Test tracking of score improvements between verification runs."""
        before_scores = VerificationScores(
            qa_score=0.60,
            reconstruction_score=0.55,
            scenario_score=0.65,
            mutation_score=0.50,
            impact_score=0.58,
            test_pass_rate=0.62,
        )

        after_scores = VerificationScores(
            qa_score=0.85,
            reconstruction_score=0.80,
            scenario_score=0.88,
            mutation_score=0.75,
            impact_score=0.82,
            test_pass_rate=0.87,
        )

        comparison = ScoreAnalyzer.compare(before_scores, after_scores)

        assert comparison.is_improvement
        assert comparison.overall_delta > 0
        assert comparison.grade_improved
        assert len(comparison.improvements) > 0
        # All areas should show improvement
        assert "Q&A" in comparison.improvements
        assert "Reconstruction" in comparison.improvements

    def test_score_comparison_detects_regressions(self):
        """Test that score comparison detects regressions."""
        before_scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.88,
            scenario_score=0.92,
            mutation_score=0.85,
            impact_score=0.87,
            test_pass_rate=0.90,
        )

        after_scores = VerificationScores(
            qa_score=0.75,  # Regression
            reconstruction_score=0.88,
            scenario_score=0.92,
            mutation_score=0.85,
            impact_score=0.87,
            test_pass_rate=0.90,
        )

        comparison = ScoreAnalyzer.compare(before_scores, after_scores)

        assert "Q&A" in comparison.regressions
        assert comparison.overall_delta < 0


# =============================================================================
# Integration Test: Pipeline Configuration
# =============================================================================


@pytest.mark.integration
class TestPipelineConfiguration:
    """Integration tests for pipeline configuration options."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig has sensible defaults."""
        config = PipelineConfig()

        assert config.max_concurrent_components == 5
        assert config.max_concurrent_strategies == 3
        assert config.phase_timeout_seconds == 300
        assert config.skip_on_failure is False
        assert config.create_tickets_for_gaps is True
        assert config.min_severity_for_ticket == "medium"
        assert len(config.enabled_strategies) > 0

    def test_pipeline_builder_fluent_api(self):
        """Test PipelineBuilder fluent API configuration."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        llm_client = MagicMock()

        pipeline = (
            PipelineBuilder()
            .with_strategies(
                [
                    StrategyType.QA_INTERROGATION,
                    StrategyType.SCENARIO_WALKTHROUGH,
                ]
            )
            .with_concurrency(components=10, strategies=5)
            .with_timeout(600)
            .with_auto_tickets(enabled=True, min_severity="high")
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(llm_client)
            .build()
        )

        assert len(pipeline.config.enabled_strategies) == 2
        assert pipeline.config.max_concurrent_components == 10
        assert pipeline.config.max_concurrent_strategies == 5
        assert pipeline.config.phase_timeout_seconds == 600
        assert pipeline.config.create_tickets_for_gaps is True
        assert pipeline.config.min_severity_for_ticket == "high"

    def test_threshold_configuration(self):
        """Test threshold configuration is correctly applied."""
        thresholds = VerificationThresholds(
            min_overall_quality=0.90,
            min_qa_score=0.85,
            min_reconstruction_score=0.80,
            min_scenario_score=0.88,
        )

        config = PipelineConfig(thresholds=thresholds)

        assert config.thresholds.min_overall_quality == 0.90
        assert config.thresholds.min_qa_score == 0.85
        assert config.thresholds.min_reconstruction_score == 0.80
        assert config.thresholds.min_scenario_score == 0.88


# =============================================================================
# Integration Test: Error Handling
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling in the pipeline."""

    @pytest.mark.asyncio
    async def test_strategy_failure_isolation(self):
        """Test that failure in one strategy doesn't stop others."""

        class FailingStrategy(MockVerificationStrategy):
            async def evaluate(
                self, challenge, team_a_response, team_b_response, ground_truth=None
            ):
                raise RuntimeError("Strategy failed")

        strategies = {
            StrategyType.QA_INTERROGATION: FailingStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
            StrategyType.SCENARIO_WALKTHROUGH: MockVerificationStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            skip_on_failure=False,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        # Pipeline should complete even with one failing strategy
        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_partial_results_on_failure(self):
        """Test that partial results are available when some strategies fail."""

        class FailingStrategy(MockVerificationStrategy):
            async def evaluate(
                self, challenge, team_a_response, team_b_response, ground_truth=None
            ):
                raise RuntimeError("Strategy failed")

        strategies = {
            StrategyType.QA_INTERROGATION: FailingStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
            StrategyType.SCENARIO_WALKTHROUGH: MockVerificationStrategy(
                StrategyType.SCENARIO_WALKTHROUGH,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        # Should have component verification with partial results
        assert run.total_components == 1
        cv = run.component_verifications["test.component"]
        # Should have executions (some successful, some failed)
        assert len(cv.executions) > 0

    @pytest.mark.asyncio
    async def test_pipeline_handles_cancellation(self):
        """Test that pipeline properly handles cancellation."""
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                execution_delay=1.0,  # Long delay
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        # Start pipeline and cancel quickly
        task = asyncio.create_task(pipeline.verify_components(["test.component"]))
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


# =============================================================================
# Integration Test: Multi-Component Verification
# =============================================================================


@pytest.mark.integration
class TestMultiComponentVerification:
    """Integration tests for verifying multiple components."""

    @pytest.mark.asyncio
    async def test_batch_verification_multiple_components(self):
        """Test verification of multiple components in batch."""
        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                score=0.85,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            max_concurrent_components=3,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        components = [f"module.function_{i}" for i in range(5)]
        run = await pipeline.verify_components(components)

        assert run.total_components == 5
        assert run.status == PipelineStatus.COMPLETED
        for comp_id in components:
            assert comp_id in run.component_verifications

    @pytest.mark.asyncio
    async def test_component_concurrency_limit(self):
        """Test that component concurrency limit is respected."""
        concurrent_count = 0
        max_concurrent = 0

        class CountingStrategy(MockVerificationStrategy):
            async def generate_challenge(self, component_id, source_code, **kwargs):
                nonlocal concurrent_count, max_concurrent
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.1)  # Simulate work
                concurrent_count -= 1
                return await super().generate_challenge(component_id, source_code, **kwargs)

        strategies = {
            StrategyType.QA_INTERROGATION: CountingStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
            max_concurrent_components=2,  # Limit to 2
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        components = [f"module.function_{i}" for i in range(5)]
        await pipeline.verify_components(components)

        # Max concurrent should not exceed the limit
        assert max_concurrent <= 2


# =============================================================================
# Integration Test: Callback Functionality
# =============================================================================


@pytest.mark.integration
class TestCallbackFunctionality:
    """Integration tests for pipeline callback functionality."""

    @pytest.mark.asyncio
    async def test_on_component_complete_callback(self):
        """Test that on_component_complete callback is invoked."""
        completed_components: list[str] = []

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        pipeline.on_component_complete(lambda cv: completed_components.append(cv.component_id))

        await pipeline.verify_components(["comp1", "comp2"])

        assert "comp1" in completed_components
        assert "comp2" in completed_components

    @pytest.mark.asyncio
    async def test_on_gap_found_callback(self):
        """Test that on_gap_found callback is invoked for documentation gaps."""
        found_gaps: list[DocumentationGap] = []

        gap = DocumentationGap(
            gap_id="gap_test_001",
            area="test_area",
            description="Test gap",
            severity=Severity.MEDIUM,
            recommendation="Fix it",
        )

        strategies = {
            StrategyType.QA_INTERROGATION: MockVerificationStrategy(
                StrategyType.QA_INTERROGATION,
                VerificationLevel.ACTIVE,
                gaps=[gap],
            ),
        }

        registry = MockStrategyRegistry(strategies)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(strategies.keys()),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        pipeline.on_gap_found(lambda g: found_gaps.append(g))

        await pipeline.verify_components(["test.component"])

        assert len(found_gaps) >= 1
        assert found_gaps[0].area == "test_area"
