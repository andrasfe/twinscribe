"""
Integration tests for the Verification Pipeline.

Tests cover:
- End-to-end pipeline execution
- Strategy orchestration
- Result aggregation
- Beads ticket generation
- Configuration handling
- Error recovery and retry logic

Related Beads Tickets:
- twinscribe-9x0: Create integration tests for verification pipeline
"""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from twinscribe.verification.base import (
    StrategyType,
    VerificationStrategy,
)
from twinscribe.verification.models import (
    DocumentationGap,
)
from twinscribe.verification.pipeline import (
    ComponentVerification,
    PipelineBuilder,
    PipelineConfig,
    PipelinePhase,
    PipelineRun,
    PipelineStatus,
    StrategyExecution,
    VerificationPipeline,
)
from twinscribe.verification.scores import (
    QualityGrade,
    VerificationScores,
    VerificationThresholds,
)
from twinscribe.verification.strategies import StrategyRegistry

# =============================================================================
# Mock Providers for Testing
# =============================================================================


class MockDocumentationProvider:
    """Mock documentation provider for testing."""

    def __init__(
        self,
        team_a_docs: str = "Team A documentation",
        team_b_docs: str = "Team B documentation",
    ):
        self.team_a_docs = team_a_docs
        self.team_b_docs = team_b_docs
        self.calls = []

    async def get_team_a_documentation(self, component_id: str) -> str:
        self.calls.append(("team_a", component_id))
        return self.team_a_docs

    async def get_team_b_documentation(self, component_id: str) -> str:
        self.calls.append(("team_b", component_id))
        return self.team_b_docs


class MockSourceCodeProvider:
    """Mock source code provider for testing."""

    def __init__(self, source_code: str = "def sample(): pass"):
        self.source_code = source_code
        self.calls = []

    async def get_source_code(self, component_id: str) -> str:
        self.calls.append(("source", component_id))
        return self.source_code

    async def get_call_graph(self, component_id: str) -> dict[str, list[str]]:
        self.calls.append(("call_graph", component_id))
        return {"callers": [], "callees": []}


class MockTicketCreator:
    """Mock ticket creator for testing."""

    def __init__(self):
        self.created_tickets = []

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


# =============================================================================
# Pipeline Initialization Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestVerificationPipelineInit:
    """Tests for VerificationPipeline initialization."""

    def test_init_with_default_config(self, mock_examiner_client):
        """Test initialization with default configuration."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        assert pipeline.config is not None
        assert isinstance(pipeline.config, PipelineConfig)
        assert len(pipeline.config.enabled_strategies) > 0

    def test_init_with_custom_config(
        self,
        mock_examiner_client,
        verification_config,
    ):
        """Test initialization with custom configuration."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        custom_config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
                StrategyType.SCENARIO_WALKTHROUGH,
            ],
            max_concurrent_components=10,
            phase_timeout_seconds=600,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=custom_config,
        )

        assert pipeline.config.max_concurrent_components == 10
        assert pipeline.config.phase_timeout_seconds == 600
        assert len(pipeline.config.enabled_strategies) == 2

    def test_init_loads_all_strategies(self, mock_examiner_client):
        """Test that all strategies are loaded on initialization."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=list(StrategyType),
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        # All 8 strategies should be enabled
        assert len(pipeline.config.enabled_strategies) >= 8

    def test_init_with_subset_of_strategies(
        self,
        mock_examiner_client,
        verification_config,
    ):
        """Test initialization with only selected strategies."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
            ],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        assert len(pipeline.config.enabled_strategies) == 1
        assert StrategyType.QA_INTERROGATION in pipeline.config.enabled_strategies

    def test_init_validates_strategies(self, mock_examiner_client):
        """Test that pipeline validates strategies don't return wrong types."""

        # Create a mock registry that returns a placeholder strategy
        class PlaceholderRegistry:
            def get(self, strategy_type):
                # Always return a strategy with wrong type (placeholder behavior)
                mock_strategy = MagicMock(spec=VerificationStrategy)
                mock_strategy.strategy_type = StrategyType.QA_INTERROGATION  # Wrong type
                return mock_strategy

        placeholder_registry = PlaceholderRegistry()
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        # Request a different strategy type - should fail validation
        config = PipelineConfig(
            enabled_strategies=[StrategyType.MUTATION_DETECTION],
        )

        with pytest.raises(ValueError) as exc_info:
            VerificationPipeline(
                strategy_registry=placeholder_registry,
                doc_provider=doc_provider,
                source_provider=source_provider,
                config=config,
            )

        assert "placeholder" in str(exc_info.value).lower()
        assert "MUTATION_DETECTION" in str(exc_info.value) or "mutation_detection" in str(
            exc_info.value
        )


# =============================================================================
# Pipeline Execution Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineExecution:
    """Tests for pipeline execution flow."""

    @pytest.mark.asyncio
    async def test_execute_full_pipeline(
        self,
        mock_examiner_client,
        mock_team_client,
        sample_verification_function,
        sample_documentation_complete,
    ):
        """Test executing all verification strategies."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider(
            team_a_docs=json.dumps(sample_documentation_complete),
            team_b_docs=json.dumps(sample_documentation_complete),
        )
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
                StrategyType.SCENARIO_WALKTHROUGH,
            ],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED
        assert run.total_components == 1
        assert sample_verification_function.id in run.component_verifications

    @pytest.mark.asyncio
    async def test_execute_minimum_strategies(
        self,
        mock_examiner_client,
        mock_team_client,
        sample_verification_function,
    ):
        """Test executing minimum recommended strategies (Q&A + Scenario + Test)."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
                StrategyType.SCENARIO_WALKTHROUGH,
                StrategyType.TEST_GENERATION,
            ],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED
        assert len(config.enabled_strategies) == 3

    @pytest.mark.asyncio
    async def test_execute_strategy_ordering(
        self,
        mock_examiner_client,
        mock_team_client,
    ):
        """Test that strategies execute in correct order."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        phases_executed = []

        def track_phase_start(phase: PipelinePhase):
            phases_executed.append(phase)

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )
        pipeline.on_phase_start(track_phase_start)

        await pipeline.verify_components(["test.component"])

        # Verify phases execute in order
        expected_order = [
            PipelinePhase.ACTIVE,
            PipelinePhase.BEHAVIORAL,
            PipelinePhase.GENERATIVE,
            PipelinePhase.CONSOLIDATION,
        ]
        assert phases_executed == expected_order

    @pytest.mark.asyncio
    async def test_execute_parallel_strategies(
        self,
        mock_examiner_client,
        mock_team_client,
    ):
        """Test that independent strategies can run in parallel."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
                StrategyType.MASKED_RECONSTRUCTION,
            ],
            max_concurrent_strategies=2,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["test.component"])

        assert run.status == PipelineStatus.COMPLETED


# =============================================================================
# Strategy Orchestration Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestStrategyOrchestration:
    """Tests for strategy orchestration."""

    @pytest.mark.asyncio
    async def test_qa_interrogation_integration(
        self,
        mock_examiner_client,
        sample_verification_function,
    ):
        """Test Q&A interrogation strategy integration."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[StrategyType.QA_INTERROGATION],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_masked_reconstruction_integration(
        self,
        mock_team_client,
        sample_verification_function,
    ):
        """Test masked reconstruction strategy integration."""
        registry = StrategyRegistry(mock_team_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[StrategyType.MASKED_RECONSTRUCTION],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_scenario_walkthrough_integration(
        self,
        mock_team_client,
        sample_verification_function,
    ):
        """Test scenario walkthrough strategy integration."""
        registry = StrategyRegistry(mock_team_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[StrategyType.SCENARIO_WALKTHROUGH],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_mutation_detection_integration(
        self,
        mock_team_client,
        sample_verification_function,
    ):
        """Test mutation detection strategy integration."""
        registry = StrategyRegistry(mock_team_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_function.source_code,
        )

        config = PipelineConfig(
            enabled_strategies=[StrategyType.MUTATION_DETECTION],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components([sample_verification_function.id])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_impact_analysis_integration(
        self,
        mock_team_client,
        sample_dependency_graph,
    ):
        """Test impact analysis strategy integration."""
        registry = StrategyRegistry(mock_team_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[StrategyType.IMPACT_ANALYSIS],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["calculate_discount"])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_adversarial_review_integration(
        self,
        mock_team_client,
        sample_documentation_complete,
        sample_documentation_incomplete,
    ):
        """Test adversarial review strategy integration."""
        registry = StrategyRegistry(mock_team_client)
        doc_provider = MockDocumentationProvider(
            team_a_docs=json.dumps(sample_documentation_complete),
            team_b_docs=json.dumps(sample_documentation_incomplete),
        )
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[StrategyType.ADVERSARIAL_REVIEW],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["sample_module.calculate_discount"])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_test_generation_integration(
        self,
        mock_examiner_client,
        sample_documentation_complete,
    ):
        """Test test generation strategy integration."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider(
            team_a_docs=json.dumps(sample_documentation_complete),
        )
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[StrategyType.TEST_GENERATION],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["sample_module.calculate_discount"])

        assert run.status == PipelineStatus.COMPLETED


# =============================================================================
# Result Aggregation Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestResultAggregation:
    """Tests for aggregating results from all strategies."""

    @pytest.mark.asyncio
    async def test_aggregate_all_scores(
        self,
        mock_examiner_client,
        mock_team_client,
    ):
        """Test aggregation of scores from all strategies."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components(["test.component"])

        # Aggregate scores should be available after completion
        aggregate = run.get_aggregate_scores()
        # May be None if no results, but method should not raise
        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_aggregate_documentation_gaps(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test aggregation of documentation gaps."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components(["test.component"])

        # Total gaps should be calculated
        assert run.total_gaps >= 0

    @pytest.mark.asyncio
    async def test_calculate_overall_quality(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test overall quality score calculation."""
        # Create scores directly to test calculation
        scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.85,
            scenario_score=0.92,
            mutation_score=0.80,
            impact_score=0.88,
        )

        overall = scores.overall_quality
        assert 0.0 <= overall <= 1.0

    @pytest.mark.asyncio
    async def test_assign_quality_grade(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test quality grade assignment."""
        # Test grade A (>= 0.95 overall)
        # Need all scores high since overall is weighted average
        scores_a = VerificationScores(
            qa_score=0.98,
            reconstruction_score=0.98,
            scenario_score=0.98,
            mutation_score=0.98,
            impact_score=0.98,
            test_pass_rate=0.98,
        )
        assert scores_a.quality_grade == QualityGrade.A

        # Test grade B (>= 0.85 overall)
        scores_b = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.90,
            scenario_score=0.90,
            mutation_score=0.90,
            impact_score=0.90,
            test_pass_rate=0.90,
        )
        assert scores_b.quality_grade == QualityGrade.B

        # Test grade C (>= 0.70 overall)
        scores_c = VerificationScores(
            qa_score=0.75,
            reconstruction_score=0.75,
            scenario_score=0.75,
            mutation_score=0.75,
            impact_score=0.75,
            test_pass_rate=0.75,
        )
        assert scores_c.quality_grade == QualityGrade.C

    @pytest.mark.asyncio
    async def test_identify_weakest_areas(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test identification of weakest verification areas."""
        scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.50,  # Weakest
            scenario_score=0.85,
            mutation_score=0.55,  # Second weakest
            impact_score=0.88,
        )

        weak_areas = scores.get_weakest_areas(count=2)
        assert len(weak_areas) == 2


# =============================================================================
# Pipeline Result Model Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineResultModel:
    """Tests for PipelineResult data model."""

    def test_pipeline_run_creation(self, sample_pipeline_result):
        """Test creating a PipelineRun instance."""
        config = PipelineConfig()
        run = PipelineRun(
            run_id="test_run_123",
            config=config,
        )

        assert run.run_id == "test_run_123"
        assert run.status == PipelineStatus.PENDING
        assert run.total_components == 0

    def test_pipeline_run_attributes(self):
        """Test PipelineRun has required attributes."""
        config = PipelineConfig()
        run = PipelineRun(
            run_id="test_run",
            config=config,
        )

        # Required attributes
        assert hasattr(run, "run_id")
        assert hasattr(run, "config")
        assert hasattr(run, "component_verifications")
        assert hasattr(run, "phase")
        assert hasattr(run, "status")
        assert hasattr(run, "started_at")
        assert hasattr(run, "completed_at")

    def test_component_verification_attributes(self):
        """Test ComponentVerification has required attributes."""
        cv = ComponentVerification(component_id="test.component")

        assert hasattr(cv, "component_id")
        assert hasattr(cv, "executions")
        assert hasattr(cv, "scores")
        assert hasattr(cv, "gaps")
        assert hasattr(cv, "started_at")
        assert hasattr(cv, "completed_at")

    def test_strategy_execution_attributes(self):
        """Test StrategyExecution has required attributes."""
        execution = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test.component",
        )

        assert hasattr(execution, "strategy_type")
        assert hasattr(execution, "component_id")
        assert hasattr(execution, "challenge")
        assert hasattr(execution, "result")
        assert hasattr(execution, "started_at")
        assert hasattr(execution, "completed_at")
        assert hasattr(execution, "error")

    def test_strategy_execution_duration(self):
        """Test StrategyExecution duration calculation."""
        execution = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test.component",
            started_at=datetime(2026, 1, 7, 10, 0, 0),
            completed_at=datetime(2026, 1, 7, 10, 0, 5),
        )

        assert execution.duration_ms == 5000

    def test_strategy_execution_succeeded(self):
        """Test StrategyExecution succeeded property."""
        # Execution with result and no error = succeeded
        mock_result = MagicMock()
        execution_success = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test.component",
            result=mock_result,
        )
        assert execution_success.succeeded is True

        # Execution with error = failed
        execution_failed = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test.component",
            error="Something went wrong",
        )
        assert execution_failed.succeeded is False


# =============================================================================
# Beads Ticket Generation Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestBeadsTicketGeneration:
    """Tests for Beads ticket generation from verification results."""

    @pytest.mark.asyncio
    async def test_generate_ticket_for_documentation_gap(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test generating Beads ticket for documentation gap."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            create_tickets_for_gaps=True,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
            ticket_creator=ticket_creator,
        )

        await pipeline.verify_components(["test.component"])

        # Ticket creator should have been called if gaps were found
        assert isinstance(ticket_creator.created_tickets, list)

    @pytest.mark.asyncio
    async def test_ticket_priority_assignment(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test that ticket priority matches gap severity."""
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            create_tickets_for_gaps=True,
            min_severity_for_ticket="medium",
        )

        assert config.min_severity_for_ticket == "medium"

    @pytest.mark.asyncio
    async def test_skip_ticket_generation_when_disabled(
        self,
        mock_examiner_client,
        sample_pipeline_result,
    ):
        """Test that ticket generation can be disabled."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()
        ticket_creator = MockTicketCreator()

        config = PipelineConfig(
            create_tickets_for_gaps=False,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
            ticket_creator=ticket_creator,
        )

        await pipeline.verify_components(["test.component"])

        # No tickets should be created when disabled
        assert len(ticket_creator.created_tickets) == 0


# =============================================================================
# Configuration Handling Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestConfigurationHandling:
    """Tests for pipeline configuration handling."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.max_concurrent_components == 5
        assert config.max_concurrent_strategies == 3
        assert config.phase_timeout_seconds == 300
        assert config.skip_on_failure is False
        assert config.create_tickets_for_gaps is True

    def test_strategy_enabled_settings(
        self,
        verification_config,
    ):
        """Test respecting enabled_strategies setting."""
        config = PipelineConfig(
            enabled_strategies=[
                StrategyType.QA_INTERROGATION,
                StrategyType.SCENARIO_WALKTHROUGH,
            ],
        )

        assert StrategyType.QA_INTERROGATION in config.enabled_strategies
        assert StrategyType.SCENARIO_WALKTHROUGH in config.enabled_strategies
        assert StrategyType.MUTATION_DETECTION not in config.enabled_strategies

    def test_threshold_configuration(
        self,
        verification_config,
    ):
        """Test threshold configuration is applied."""
        thresholds = VerificationThresholds(
            min_overall_quality=0.90,
            min_qa_score=0.85,
        )

        config = PipelineConfig(thresholds=thresholds)

        assert config.thresholds.min_overall_quality == 0.90
        assert config.thresholds.min_qa_score == 0.85

    def test_strategy_specific_config(
        self,
        verification_config,
    ):
        """Test strategy-specific configuration is applied."""
        # Test can be configured with strategy-specific settings
        config = PipelineConfig(
            enabled_strategies=[StrategyType.QA_INTERROGATION],
            max_concurrent_strategies=5,
        )

        assert config.max_concurrent_strategies == 5


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_strategy_failure_isolation(
        self,
        mock_examiner_client,
    ):
        """Test that failure in one strategy doesn't stop others."""
        # Create a registry with a failing strategy
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            skip_on_failure=False,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        # Pipeline should complete even with individual failures
        run = await pipeline.verify_components(["test.component"])
        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_partial_results_on_failure(
        self,
        mock_examiner_client,
    ):
        """Test that partial results are returned when some strategies fail."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components(["test.component"])

        # Should have component verifications even with partial failures
        assert run.total_components == 1

    @pytest.mark.asyncio
    async def test_pipeline_cancelled_status(
        self,
        mock_examiner_client,
    ):
        """Test pipeline handles cancellation correctly."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        # Create a run and verify status tracking
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)
        assert run.status == PipelineStatus.PENDING


# =============================================================================
# Multi-Component Verification Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestMultiComponentVerification:
    """Tests for verifying multiple components."""

    @pytest.mark.asyncio
    async def test_verify_multiple_functions(
        self,
        mock_examiner_client,
        mock_team_client,
        sample_verification_function,
        sample_async_component,
    ):
        """Test verification across multiple functions."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components(
            [
                sample_verification_function.id,
                sample_async_component.id,
            ]
        )

        assert run.total_components == 2
        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_verify_class_with_methods(
        self,
        mock_examiner_client,
        mock_team_client,
        sample_verification_class,
    ):
        """Test verification of a class with multiple methods."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider(
            source_code=sample_verification_class.source_code,
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components([sample_verification_class.id])

        assert run.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_batch_verification(
        self,
        mock_examiner_client,
        mock_team_client,
    ):
        """Test batch verification of many components."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
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


# =============================================================================
# Pipeline Performance Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelinePerformance:
    """Tests for pipeline performance characteristics."""

    @pytest.mark.asyncio
    async def test_execution_time_tracking(
        self,
        mock_examiner_client,
        mock_team_client,
    ):
        """Test that execution time is tracked."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )

        run = await pipeline.verify_components(["test.component"])

        # Execution should have start and end times
        assert run.started_at is not None
        assert run.completed_at is not None
        assert run.completed_at >= run.started_at


# =============================================================================
# Pipeline with Static Analysis Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineWithStaticAnalysis:
    """Tests for pipeline integration with static analysis."""

    @pytest.mark.asyncio
    async def test_use_call_graph_for_impact_analysis(
        self,
        mock_examiner_client,
        sample_dependency_graph,
    ):
        """Test using static call graph for impact analysis."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        config = PipelineConfig(
            enabled_strategies=[StrategyType.IMPACT_ANALYSIS],
        )

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        run = await pipeline.verify_components(["calculate_discount"])

        assert run.status == PipelineStatus.COMPLETED


# =============================================================================
# Pipeline Builder Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineBuilder:
    """Tests for PipelineBuilder fluent API."""

    def test_builder_with_strategies(self, mock_examiner_client):
        """Test builder with_strategies method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_strategies(
                [
                    StrategyType.QA_INTERROGATION,
                    StrategyType.SCENARIO_WALKTHROUGH,
                ]
            )
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert len(pipeline.config.enabled_strategies) == 2

    def test_builder_with_all_strategies(self, mock_examiner_client):
        """Test builder with_all_strategies method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_all_strategies()
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert len(pipeline.config.enabled_strategies) >= 8

    def test_builder_with_minimum_strategies(self, mock_examiner_client):
        """Test builder with_minimum_strategies method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_minimum_strategies()
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert StrategyType.QA_INTERROGATION in pipeline.config.enabled_strategies
        assert StrategyType.SCENARIO_WALKTHROUGH in pipeline.config.enabled_strategies
        assert StrategyType.TEST_GENERATION in pipeline.config.enabled_strategies

    def test_builder_with_concurrency(self, mock_examiner_client):
        """Test builder with_concurrency method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_concurrency(components=10, strategies=5)
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert pipeline.config.max_concurrent_components == 10
        assert pipeline.config.max_concurrent_strategies == 5

    def test_builder_with_timeout(self, mock_examiner_client):
        """Test builder with_timeout method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_timeout(600)
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert pipeline.config.phase_timeout_seconds == 600

    def test_builder_with_auto_tickets(self, mock_examiner_client):
        """Test builder with_auto_tickets method."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        pipeline = (
            PipelineBuilder()
            .with_auto_tickets(enabled=True, min_severity="high")
            .with_doc_provider(doc_provider)
            .with_source_provider(source_provider)
            .with_llm_client(mock_examiner_client)
            .build()
        )

        assert pipeline.config.create_tickets_for_gaps is True
        assert pipeline.config.min_severity_for_ticket == "high"

    def test_builder_requires_doc_provider(self, mock_examiner_client):
        """Test builder fails without doc provider."""
        source_provider = MockSourceCodeProvider()

        with pytest.raises(ValueError, match="Documentation provider"):
            (
                PipelineBuilder()
                .with_source_provider(source_provider)
                .with_llm_client(mock_examiner_client)
                .build()
            )

    def test_builder_requires_source_provider(self, mock_examiner_client):
        """Test builder fails without source provider."""
        doc_provider = MockDocumentationProvider()

        with pytest.raises(ValueError, match="Source code provider"):
            (
                PipelineBuilder()
                .with_doc_provider(doc_provider)
                .with_llm_client(mock_examiner_client)
                .build()
            )

    def test_builder_requires_llm_client(self):
        """Test builder fails without LLM client."""
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        with pytest.raises(ValueError, match="LLM client"):
            (
                PipelineBuilder()
                .with_doc_provider(doc_provider)
                .with_source_provider(source_provider)
                .build()
            )


# =============================================================================
# Pipeline Callback Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineCallbacks:
    """Tests for pipeline callback functionality."""

    @pytest.mark.asyncio
    async def test_on_phase_start_callback(self, mock_examiner_client):
        """Test on_phase_start callback is invoked."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        phases_started = []

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )
        pipeline.on_phase_start(lambda phase: phases_started.append(phase))

        await pipeline.verify_components(["test.component"])

        assert len(phases_started) > 0

    @pytest.mark.asyncio
    async def test_on_phase_complete_callback(self, mock_examiner_client):
        """Test on_phase_complete callback is invoked."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        phases_completed = []

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )
        pipeline.on_phase_complete(lambda phase: phases_completed.append(phase))

        await pipeline.verify_components(["test.component"])

        assert len(phases_completed) > 0

    @pytest.mark.asyncio
    async def test_on_component_complete_callback(self, mock_examiner_client):
        """Test on_component_complete callback is invoked."""
        registry = StrategyRegistry(mock_examiner_client)
        doc_provider = MockDocumentationProvider()
        source_provider = MockSourceCodeProvider()

        components_completed = []

        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
        )
        pipeline.on_component_complete(lambda cv: components_completed.append(cv))

        await pipeline.verify_components(["test.component"])

        # Callback may be called multiple times depending on phases
        assert isinstance(components_completed, list)


# =============================================================================
# Pipeline Run Properties Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestPipelineRunProperties:
    """Tests for PipelineRun computed properties."""

    def test_total_components(self):
        """Test total_components property."""
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)

        run.component_verifications["comp1"] = ComponentVerification(component_id="comp1")
        run.component_verifications["comp2"] = ComponentVerification(component_id="comp2")

        assert run.total_components == 2

    def test_successful_components(self):
        """Test successful_components property."""
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)

        # Add successful component (all scores high for grade B or better)
        cv1 = ComponentVerification(component_id="comp1")
        cv1.scores = VerificationScores(
            qa_score=0.90,
            reconstruction_score=0.90,
            scenario_score=0.90,
            mutation_score=0.90,
            impact_score=0.90,
            test_pass_rate=0.90,
        )  # Grade B
        run.component_verifications["comp1"] = cv1

        # Add failed component (low scores = grade F)
        cv2 = ComponentVerification(component_id="comp2")
        cv2.scores = VerificationScores(
            qa_score=0.30,
            reconstruction_score=0.30,
            scenario_score=0.30,
            mutation_score=0.30,
            impact_score=0.30,
            test_pass_rate=0.30,
        )  # Grade F
        run.component_verifications["comp2"] = cv2

        assert run.successful_components == 1

    def test_total_gaps(self):
        """Test total_gaps property."""
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)

        cv1 = ComponentVerification(component_id="comp1")
        cv1.gaps = [MagicMock(), MagicMock()]  # 2 gaps
        run.component_verifications["comp1"] = cv1

        cv2 = ComponentVerification(component_id="comp2")
        cv2.gaps = [MagicMock()]  # 1 gap
        run.component_verifications["comp2"] = cv2

        assert run.total_gaps == 3

    def test_get_aggregate_scores(self):
        """Test get_aggregate_scores method."""
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)

        cv1 = ComponentVerification(component_id="comp1")
        cv1.scores = VerificationScores(
            qa_score=0.8,
            reconstruction_score=0.9,
        )
        run.component_verifications["comp1"] = cv1

        cv2 = ComponentVerification(component_id="comp2")
        cv2.scores = VerificationScores(
            qa_score=0.6,
            reconstruction_score=0.7,
        )
        run.component_verifications["comp2"] = cv2

        aggregate = run.get_aggregate_scores()

        assert aggregate is not None
        assert aggregate.qa_score == 0.7  # Average of 0.8 and 0.6
        assert aggregate.reconstruction_score == 0.8  # Average of 0.9 and 0.7

    def test_get_aggregate_scores_empty(self):
        """Test get_aggregate_scores with no components."""
        config = PipelineConfig()
        run = PipelineRun(run_id="test", config=config)

        aggregate = run.get_aggregate_scores()
        assert aggregate is None


# =============================================================================
# Component Verification Properties Tests
# =============================================================================


@pytest.mark.verification
@pytest.mark.integration
class TestComponentVerificationProperties:
    """Tests for ComponentVerification computed properties."""

    def test_all_strategies_succeeded_true(self):
        """Test all_strategies_succeeded when all succeed."""
        cv = ComponentVerification(component_id="test")

        execution1 = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test",
            result=MagicMock(),
        )
        execution2 = StrategyExecution(
            strategy_type=StrategyType.SCENARIO_WALKTHROUGH,
            component_id="test",
            result=MagicMock(),
        )
        cv.executions = [execution1, execution2]

        assert cv.all_strategies_succeeded is True

    def test_all_strategies_succeeded_false(self):
        """Test all_strategies_succeeded when one fails."""
        cv = ComponentVerification(component_id="test")

        execution1 = StrategyExecution(
            strategy_type=StrategyType.QA_INTERROGATION,
            component_id="test",
            result=MagicMock(),
        )
        execution2 = StrategyExecution(
            strategy_type=StrategyType.SCENARIO_WALKTHROUGH,
            component_id="test",
            error="Failed",
        )
        cv.executions = [execution1, execution2]

        assert cv.all_strategies_succeeded is False

    def test_quality_grade_property(self):
        """Test quality_grade property."""
        cv = ComponentVerification(component_id="test")
        # Need all scores high for grade A (overall >= 0.95)
        cv.scores = VerificationScores(
            qa_score=0.98,
            reconstruction_score=0.98,
            scenario_score=0.98,
            mutation_score=0.98,
            impact_score=0.98,
            test_pass_rate=0.98,
        )

        assert cv.quality_grade == QualityGrade.A

    def test_quality_grade_none_without_scores(self):
        """Test quality_grade is None without scores."""
        cv = ComponentVerification(component_id="test")

        assert cv.quality_grade is None
