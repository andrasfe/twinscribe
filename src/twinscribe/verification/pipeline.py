"""
CrossCheck Verification Framework - Pipeline Integration.

This module provides the VerificationPipeline orchestrator that integrates
verification strategies with the existing dual-stream documentation pipeline.

The pipeline operates in phases:
1. PASSIVE - Basic comparison (existing)
2. ACTIVE - Q&A, masked reconstruction, scenario walkthrough
3. BEHAVIORAL - Mutation detection, impact analysis
4. GENERATIVE - Code reconstruction, test generation

The pipeline can be configured to run all strategies or a subset,
with configurable thresholds and parallelization options.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from twinscribe.verification.base import (
    StrategyType,
    VerificationLevel,
    VerificationStrategy,
)
from twinscribe.verification.models import (
    DocumentationGap,
    VerificationChallenge,
    VerificationResult,
)
from twinscribe.verification.scores import (
    QualityGrade,
    VerificationScores,
    VerificationThresholds,
)
from twinscribe.verification.strategies import StrategyRegistry


class PipelinePhase(str, Enum):
    """Phases of the verification pipeline."""

    PASSIVE = "passive"
    ACTIVE = "active"
    BEHAVIORAL = "behavioral"
    GENERATIVE = "generative"
    CONSOLIDATION = "consolidation"


class PipelineStatus(str, Enum):
    """Status of pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentationProvider(Protocol):
    """Protocol for documentation providers.

    Implementations fetch documentation from Team A and Team B
    for a given component.
    """

    async def get_team_a_documentation(self, component_id: str) -> str:
        """Get Team A's documentation for a component."""
        ...

    async def get_team_b_documentation(self, component_id: str) -> str:
        """Get Team B's documentation for a component."""
        ...


class SourceCodeProvider(Protocol):
    """Protocol for source code providers.

    Implementations fetch source code for components.
    """

    async def get_source_code(self, component_id: str) -> str:
        """Get source code for a component."""
        ...

    async def get_call_graph(self, component_id: str) -> dict[str, list[str]]:
        """Get call graph (callers and callees) for static analysis."""
        ...


class TicketCreator(Protocol):
    """Protocol for creating tickets from documentation gaps.

    Implementations create Beads tickets for gaps that need human review.
    """

    async def create_gap_ticket(
        self,
        gap: DocumentationGap,
        component_id: str,
        strategy_type: StrategyType,
    ) -> str:
        """Create a ticket for a documentation gap.

        Returns:
            Ticket key/ID
        """
        ...


class PipelineConfig(BaseModel):
    """Configuration for the verification pipeline.

    Attributes:
        enabled_strategies: Which strategies to run
        thresholds: Quality thresholds for pass/fail
        max_concurrent_components: Max components to verify in parallel
        max_concurrent_strategies: Max strategies to run in parallel
        phase_timeout_seconds: Timeout for each phase
        skip_on_failure: Whether to skip remaining strategies on failure
        create_tickets_for_gaps: Whether to auto-create tickets for gaps
        min_severity_for_ticket: Minimum severity for ticket creation
    """

    enabled_strategies: list[StrategyType] = Field(
        default_factory=lambda: [
            StrategyType.QA_INTERROGATION,
            StrategyType.MASKED_RECONSTRUCTION,
            StrategyType.SCENARIO_WALKTHROUGH,
            StrategyType.MUTATION_DETECTION,
            StrategyType.IMPACT_ANALYSIS,
        ],
        description="Strategies to run",
    )
    thresholds: VerificationThresholds = Field(
        default_factory=VerificationThresholds,
        description="Quality thresholds",
    )
    max_concurrent_components: int = Field(
        default=5,
        ge=1,
        description="Max parallel components",
    )
    max_concurrent_strategies: int = Field(
        default=3,
        ge=1,
        description="Max parallel strategies",
    )
    phase_timeout_seconds: int = Field(
        default=300,
        ge=30,
        description="Phase timeout",
    )
    skip_on_failure: bool = Field(
        default=False,
        description="Skip remaining on failure",
    )
    create_tickets_for_gaps: bool = Field(
        default=True,
        description="Auto-create tickets for gaps",
    )
    min_severity_for_ticket: str = Field(
        default="medium",
        description="Min severity for tickets",
    )


@dataclass
class StrategyExecution:
    """Record of a single strategy execution.

    Attributes:
        strategy_type: Strategy that was executed
        component_id: Component that was verified
        challenge: The generated challenge
        result: The evaluation result
        started_at: When execution started
        completed_at: When execution completed
        error: Error message if failed
    """

    strategy_type: StrategyType
    component_id: str
    challenge: VerificationChallenge | None = None
    result: VerificationResult | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        """Execution duration in milliseconds."""
        if not self.started_at or not self.completed_at:
            return 0
        delta = self.completed_at - self.started_at
        return int(delta.total_seconds() * 1000)

    @property
    def succeeded(self) -> bool:
        """Whether execution succeeded."""
        return self.result is not None and self.error is None


@dataclass
class ComponentVerification:
    """Verification results for a single component.

    Attributes:
        component_id: Component identifier
        executions: Strategy execution records
        scores: Aggregated verification scores
        gaps: All documentation gaps found
        started_at: When verification started
        completed_at: When verification completed
    """

    component_id: str
    executions: list[StrategyExecution] = field(default_factory=list)
    scores: VerificationScores | None = None
    gaps: list[DocumentationGap] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def all_strategies_succeeded(self) -> bool:
        """Whether all strategies succeeded."""
        return all(e.succeeded for e in self.executions)

    @property
    def quality_grade(self) -> QualityGrade | None:
        """Quality grade if scores are available."""
        return self.scores.quality_grade if self.scores else None


@dataclass
class PipelineRun:
    """Record of a complete pipeline run.

    Attributes:
        run_id: Unique identifier for this run
        config: Pipeline configuration used
        component_verifications: Results per component
        phase: Current phase
        status: Current status
        started_at: When run started
        completed_at: When run completed
        error: Error message if failed
    """

    run_id: str
    config: PipelineConfig
    component_verifications: dict[str, ComponentVerification] = field(default_factory=dict)
    phase: PipelinePhase = PipelinePhase.PASSIVE
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def total_components(self) -> int:
        """Total number of components verified."""
        return len(self.component_verifications)

    @property
    def successful_components(self) -> int:
        """Number of components that passed verification."""
        return sum(
            1
            for cv in self.component_verifications.values()
            if cv.scores and cv.scores.quality_grade != QualityGrade.F
        )

    @property
    def total_gaps(self) -> int:
        """Total documentation gaps found."""
        return sum(len(cv.gaps) for cv in self.component_verifications.values())

    def get_aggregate_scores(self) -> VerificationScores | None:
        """Get aggregated scores across all components."""
        valid_scores = [
            cv.scores for cv in self.component_verifications.values() if cv.scores is not None
        ]
        if not valid_scores:
            return None

        # Average across components
        return VerificationScores(
            qa_score=sum(s.qa_score for s in valid_scores) / len(valid_scores),
            reconstruction_score=sum(s.reconstruction_score for s in valid_scores)
            / len(valid_scores),
            scenario_score=sum(s.scenario_score for s in valid_scores) / len(valid_scores),
            mutation_score=sum(s.mutation_score for s in valid_scores) / len(valid_scores),
            impact_score=sum(s.impact_score for s in valid_scores) / len(valid_scores),
            adversarial_findings=sum(s.adversarial_findings for s in valid_scores),
            test_pass_rate=sum(s.test_pass_rate for s in valid_scores) / len(valid_scores),
        )


class VerificationPipeline:
    """Orchestrates verification across the dual-stream pipeline.

    The VerificationPipeline integrates with the existing documentation
    system and coordinates all verification strategies. It supports:

    - Phased execution (passive -> active -> behavioral -> generative)
    - Parallel execution with configurable concurrency
    - Score aggregation and quality grading
    - Automatic ticket creation for gaps
    - Progress tracking and callbacks

    Usage:
        pipeline = VerificationPipeline(
            strategy_registry=registry,
            doc_provider=doc_provider,
            source_provider=source_provider,
            config=config,
        )

        # Verify specific components
        run = await pipeline.verify_components(["module.Class.method"])

        # Get results
        print(f"Grade: {run.get_aggregate_scores().quality_grade}")

        # Or verify after existing pipeline
        run = await pipeline.verify_after_comparison(comparison_result)
    """

    def __init__(
        self,
        strategy_registry: StrategyRegistry,
        doc_provider: DocumentationProvider,
        source_provider: SourceCodeProvider,
        config: PipelineConfig | None = None,
        ticket_creator: TicketCreator | None = None,
    ) -> None:
        """Initialize the verification pipeline.

        Args:
            strategy_registry: Registry providing strategy instances
            doc_provider: Provider for team documentation
            source_provider: Provider for source code
            config: Pipeline configuration
            ticket_creator: Optional ticket creator for gaps
        """
        self._registry = strategy_registry
        self._doc_provider = doc_provider
        self._source_provider = source_provider
        self._config = config or PipelineConfig()
        self._ticket_creator = ticket_creator

        # Callbacks
        self._on_phase_start: list[Callable[[PipelinePhase], None]] = []
        self._on_phase_complete: list[Callable[[PipelinePhase], None]] = []
        self._on_component_complete: list[Callable[[ComponentVerification], None]] = []
        self._on_gap_found: list[Callable[[DocumentationGap], None]] = []

        # Validate strategies to prevent placeholder strategies from running
        self._validate_strategies()

    def _validate_strategies(self) -> None:
        """Validate that enabled strategies are properly implemented.

        Checks that each strategy returns the expected strategy type,
        preventing placeholder strategies from silently producing incorrect results.

        Raises:
            ValueError: If a strategy returns an incorrect type (placeholder)
        """
        for strategy_type in self._config.enabled_strategies:
            strategy = self._registry.get(strategy_type)
            if strategy.strategy_type != strategy_type:
                raise ValueError(
                    f"Strategy {strategy_type.value} returns incorrect type "
                    f"'{strategy.strategy_type.value}' (appears to be a placeholder). "
                    f"Please implement the proper strategy or remove it from enabled_strategies."
                )

    @property
    def config(self) -> PipelineConfig:
        """Get the pipeline configuration."""
        return self._config

    def on_phase_start(self, callback: Callable[[PipelinePhase], None]) -> None:
        """Register callback for phase start."""
        self._on_phase_start.append(callback)

    def on_phase_complete(self, callback: Callable[[PipelinePhase], None]) -> None:
        """Register callback for phase completion."""
        self._on_phase_complete.append(callback)

    def on_component_complete(self, callback: Callable[[ComponentVerification], None]) -> None:
        """Register callback for component verification completion."""
        self._on_component_complete.append(callback)

    def on_gap_found(self, callback: Callable[[DocumentationGap], None]) -> None:
        """Register callback for gap discovery."""
        self._on_gap_found.append(callback)

    async def verify_components(
        self,
        component_ids: list[str],
        run_id: str | None = None,
    ) -> PipelineRun:
        """Verify a list of components.

        Args:
            component_ids: Components to verify
            run_id: Optional run identifier

        Returns:
            PipelineRun with all results
        """
        import uuid

        run = PipelineRun(
            run_id=run_id or f"run_{uuid.uuid4().hex[:8]}",
            config=self._config,
            started_at=datetime.utcnow(),
        )

        try:
            run.status = PipelineStatus.RUNNING

            # Execute phases
            for phase in [
                PipelinePhase.ACTIVE,
                PipelinePhase.BEHAVIORAL,
                PipelinePhase.GENERATIVE,
                PipelinePhase.CONSOLIDATION,
            ]:
                run.phase = phase
                self._notify_phase_start(phase)

                if phase == PipelinePhase.CONSOLIDATION:
                    await self._consolidation_phase(run)
                else:
                    await self._execute_phase(run, component_ids, phase)

                self._notify_phase_complete(phase)

            run.status = PipelineStatus.COMPLETED

        except asyncio.CancelledError:
            run.status = PipelineStatus.CANCELLED
            raise
        except Exception as e:
            run.status = PipelineStatus.FAILED
            run.error = str(e)
            raise
        finally:
            run.completed_at = datetime.utcnow()

        return run

    async def verify_after_comparison(
        self,
        comparison_result: Any,  # ComparisonResult from existing system
        run_id: str | None = None,
    ) -> PipelineRun:
        """Run verification after the existing comparison phase.

        This integrates with Phase 2 (Passive Comparison) of the existing
        pipeline and adds active verification.

        Args:
            comparison_result: Result from the comparator agent
            run_id: Optional run identifier

        Returns:
            PipelineRun with verification results
        """
        # Extract component IDs from comparison result
        component_ids = []
        if hasattr(comparison_result, "discrepancies"):
            component_ids = list({d.component_id for d in comparison_result.discrepancies})
        elif hasattr(comparison_result, "component_ids"):
            component_ids = comparison_result.component_ids

        return await self.verify_components(component_ids, run_id)

    async def _execute_phase(
        self,
        run: PipelineRun,
        component_ids: list[str],
        phase: PipelinePhase,
    ) -> None:
        """Execute a verification phase.

        Args:
            run: Current pipeline run
            component_ids: Components to verify
            phase: Phase to execute
        """
        # Get strategies for this phase
        level_map = {
            PipelinePhase.ACTIVE: VerificationLevel.ACTIVE,
            PipelinePhase.BEHAVIORAL: VerificationLevel.BEHAVIORAL,
            PipelinePhase.GENERATIVE: VerificationLevel.GENERATIVE,
        }
        level = level_map.get(phase)
        if not level:
            return

        strategies = [
            s
            for s in [self._registry.get(st) for st in self._config.enabled_strategies]
            if s.level == level
        ]

        if not strategies:
            return

        # Create semaphore for concurrency control
        component_semaphore = asyncio.Semaphore(self._config.max_concurrent_components)

        async def verify_component(component_id: str) -> None:
            async with component_semaphore:
                await self._verify_component(run, component_id, strategies)

        # Verify all components in parallel (with limit)
        await asyncio.gather(
            *[verify_component(cid) for cid in component_ids],
            return_exceptions=True,
        )

    async def _verify_component(
        self,
        run: PipelineRun,
        component_id: str,
        strategies: list[VerificationStrategy],
    ) -> None:
        """Verify a single component with given strategies.

        Args:
            run: Current pipeline run
            component_id: Component to verify
            strategies: Strategies to apply
        """
        # Get or create component verification
        if component_id not in run.component_verifications:
            run.component_verifications[component_id] = ComponentVerification(
                component_id=component_id,
                started_at=datetime.utcnow(),
            )
        cv = run.component_verifications[component_id]

        # Fetch required data
        source_code = await self._source_provider.get_source_code(component_id)
        team_a_docs = await self._doc_provider.get_team_a_documentation(component_id)
        team_b_docs = await self._doc_provider.get_team_b_documentation(component_id)

        # Execute strategies with concurrency limit
        strategy_semaphore = asyncio.Semaphore(self._config.max_concurrent_strategies)

        async def run_strategy(strategy: VerificationStrategy) -> StrategyExecution:
            async with strategy_semaphore:
                return await self._execute_strategy(
                    strategy,
                    component_id,
                    source_code,
                    team_a_docs,
                    team_b_docs,
                )

        executions = await asyncio.gather(
            *[run_strategy(s) for s in strategies],
            return_exceptions=True,
        )

        # Process results
        for execution in executions:
            if isinstance(execution, Exception):
                cv.executions.append(
                    StrategyExecution(
                        strategy_type=StrategyType.QA_INTERROGATION,  # Unknown
                        component_id=component_id,
                        error=str(execution),
                    )
                )
            else:
                cv.executions.append(execution)
                if execution.result:
                    cv.gaps.extend(execution.result.documentation_gaps)

        # Update component verification
        cv.completed_at = datetime.utcnow()
        self._update_component_scores(cv)

        # Notify
        self._notify_component_complete(cv)

        # Create tickets for gaps
        if self._config.create_tickets_for_gaps and self._ticket_creator:
            await self._create_tickets_for_gaps(cv)

    async def _execute_strategy(
        self,
        strategy: VerificationStrategy,
        component_id: str,
        source_code: str,
        team_a_docs: str,
        team_b_docs: str,
    ) -> StrategyExecution:
        """Execute a single strategy.

        Args:
            strategy: Strategy to execute
            component_id: Component being verified
            source_code: Component source code
            team_a_docs: Team A's documentation
            team_b_docs: Team B's documentation

        Returns:
            StrategyExecution record
        """
        execution = StrategyExecution(
            strategy_type=strategy.strategy_type,
            component_id=component_id,
            started_at=datetime.utcnow(),
        )

        try:
            # Generate challenge
            challenge = await strategy.generate_challenge(
                component_id=component_id,
                source_code=source_code,
            )
            execution.challenge = challenge

            # Teams respond to challenge using their documentation
            # In production, this would invoke the documentation agents
            team_a_response = await self._get_team_response(
                strategy.strategy_type, challenge, team_a_docs
            )
            team_b_response = await self._get_team_response(
                strategy.strategy_type, challenge, team_b_docs
            )

            # Evaluate responses
            result = await strategy.evaluate(
                challenge=challenge,
                team_a_response=team_a_response,
                team_b_response=team_b_response,
                ground_truth=source_code,
            )
            execution.result = result

            # Notify for gaps
            for gap in result.documentation_gaps:
                self._notify_gap_found(gap)

        except Exception as e:
            execution.error = str(e)

        execution.completed_at = datetime.utcnow()
        return execution

    async def _get_team_response(
        self,
        strategy_type: StrategyType,
        challenge: VerificationChallenge,
        documentation: str,
    ) -> str:
        """Get a team's response to a challenge.

        In production, this would invoke the documentation agent.
        For now, returns a placeholder response.

        Args:
            strategy_type: Type of strategy
            challenge: The challenge
            documentation: Team's documentation

        Returns:
            JSON string response
        """
        # Placeholder - in production, invoke the documentation agent
        # to respond to the challenge using only their documentation
        return "[]"

    def _update_component_scores(self, cv: ComponentVerification) -> None:
        """Update aggregated scores for a component.

        Args:
            cv: Component verification to update
        """
        # Extract scores from executions
        scores_by_type: dict[StrategyType, float] = {}

        for execution in cv.executions:
            if execution.result:
                scores_by_type[execution.strategy_type] = execution.result.average_score

        # Create VerificationScores
        cv.scores = VerificationScores(
            qa_score=scores_by_type.get(StrategyType.QA_INTERROGATION, 0.0),
            reconstruction_score=scores_by_type.get(StrategyType.MASKED_RECONSTRUCTION, 0.0),
            scenario_score=scores_by_type.get(StrategyType.SCENARIO_WALKTHROUGH, 0.0),
            mutation_score=scores_by_type.get(StrategyType.MUTATION_DETECTION, 0.0),
            impact_score=scores_by_type.get(StrategyType.IMPACT_ANALYSIS, 0.0),
            adversarial_findings=len(
                [
                    e
                    for e in cv.executions
                    if e.strategy_type == StrategyType.ADVERSARIAL_REVIEW and e.result
                ]
            ),
            test_pass_rate=scores_by_type.get(StrategyType.TEST_GENERATION, 0.0),
        )

    async def _consolidation_phase(self, run: PipelineRun) -> None:
        """Execute the consolidation phase.

        Aggregates findings, generates reports, and creates tickets.

        Args:
            run: Current pipeline run
        """
        # Aggregate all gaps
        all_gaps = []
        for cv in run.component_verifications.values():
            all_gaps.extend(cv.gaps)

        # Deduplicate gaps by area
        unique_gaps: dict[str, DocumentationGap] = {}
        for gap in all_gaps:
            key = f"{gap.area}:{gap.description}"
            if key not in unique_gaps:
                unique_gaps[key] = gap

        # In production, would generate summary reports and additional tickets

    async def _create_tickets_for_gaps(self, cv: ComponentVerification) -> None:
        """Create tickets for documentation gaps.

        Args:
            cv: Component verification with gaps
        """
        if not self._ticket_creator:
            return

        severity_order = ["critical", "high", "medium", "low"]
        min_index = severity_order.index(self._config.min_severity_for_ticket)

        for gap in cv.gaps:
            gap_index = severity_order.index(gap.severity.value)
            if gap_index <= min_index:
                # Find the strategy that found this gap
                strategy_type = StrategyType.QA_INTERROGATION  # Default
                for execution in cv.executions:
                    if execution.result and gap in execution.result.documentation_gaps:
                        strategy_type = execution.strategy_type
                        break

                await self._ticket_creator.create_gap_ticket(
                    gap=gap,
                    component_id=cv.component_id,
                    strategy_type=strategy_type,
                )

    def _notify_phase_start(self, phase: PipelinePhase) -> None:
        """Notify phase start callbacks."""
        for callback in self._on_phase_start:
            try:
                callback(phase)
            except Exception:
                pass

    def _notify_phase_complete(self, phase: PipelinePhase) -> None:
        """Notify phase complete callbacks."""
        for callback in self._on_phase_complete:
            try:
                callback(phase)
            except Exception:
                pass

    def _notify_component_complete(self, cv: ComponentVerification) -> None:
        """Notify component complete callbacks."""
        for callback in self._on_component_complete:
            try:
                callback(cv)
            except Exception:
                pass

    def _notify_gap_found(self, gap: DocumentationGap) -> None:
        """Notify gap found callbacks."""
        for callback in self._on_gap_found:
            try:
                callback(gap)
            except Exception:
                pass


class PipelineBuilder:
    """Builder for constructing VerificationPipeline instances.

    Provides a fluent interface for configuration.

    Usage:
        pipeline = (
            PipelineBuilder()
            .with_strategies([StrategyType.QA_INTERROGATION, StrategyType.SCENARIO_WALKTHROUGH])
            .with_threshold("qa", 0.8)
            .with_concurrency(components=10, strategies=5)
            .with_doc_provider(my_provider)
            .with_source_provider(my_source)
            .with_llm_client(my_llm)
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._config = PipelineConfig()
        self._doc_provider: DocumentationProvider | None = None
        self._source_provider: SourceCodeProvider | None = None
        self._ticket_creator: TicketCreator | None = None
        self._llm_client: Any = None

    def with_strategies(self, strategies: list[StrategyType]) -> "PipelineBuilder":
        """Set enabled strategies."""
        self._config.enabled_strategies = strategies
        return self

    def with_all_strategies(self) -> "PipelineBuilder":
        """Enable all strategies."""
        self._config.enabled_strategies = list(StrategyType)
        return self

    def with_minimum_strategies(self) -> "PipelineBuilder":
        """Enable recommended minimum strategies."""
        self._config.enabled_strategies = [
            StrategyType.QA_INTERROGATION,
            StrategyType.SCENARIO_WALKTHROUGH,
            StrategyType.TEST_GENERATION,
        ]
        return self

    def with_threshold(self, strategy: str, threshold: float) -> "PipelineBuilder":
        """Set a specific threshold."""
        if hasattr(self._config.thresholds, f"min_{strategy}_score"):
            setattr(self._config.thresholds, f"min_{strategy}_score", threshold)
        return self

    def with_overall_threshold(self, threshold: float) -> "PipelineBuilder":
        """Set the overall quality threshold."""
        self._config.thresholds.min_overall_quality = threshold
        return self

    def with_concurrency(self, components: int = 5, strategies: int = 3) -> "PipelineBuilder":
        """Set concurrency limits."""
        self._config.max_concurrent_components = components
        self._config.max_concurrent_strategies = strategies
        return self

    def with_timeout(self, seconds: int) -> "PipelineBuilder":
        """Set phase timeout."""
        self._config.phase_timeout_seconds = seconds
        return self

    def with_doc_provider(self, provider: DocumentationProvider) -> "PipelineBuilder":
        """Set the documentation provider."""
        self._doc_provider = provider
        return self

    def with_source_provider(self, provider: SourceCodeProvider) -> "PipelineBuilder":
        """Set the source code provider."""
        self._source_provider = provider
        return self

    def with_ticket_creator(self, creator: TicketCreator) -> "PipelineBuilder":
        """Set the ticket creator."""
        self._ticket_creator = creator
        return self

    def with_llm_client(self, client: Any) -> "PipelineBuilder":
        """Set the LLM client."""
        self._llm_client = client
        return self

    def with_auto_tickets(
        self, enabled: bool = True, min_severity: str = "medium"
    ) -> "PipelineBuilder":
        """Configure automatic ticket creation."""
        self._config.create_tickets_for_gaps = enabled
        self._config.min_severity_for_ticket = min_severity
        return self

    def build(self) -> VerificationPipeline:
        """Build the pipeline instance.

        Returns:
            Configured VerificationPipeline

        Raises:
            ValueError: If required providers are not set
        """
        if not self._doc_provider:
            raise ValueError("Documentation provider is required")
        if not self._source_provider:
            raise ValueError("Source code provider is required")
        if not self._llm_client:
            raise ValueError("LLM client is required")

        registry = StrategyRegistry(self._llm_client)

        return VerificationPipeline(
            strategy_registry=registry,
            doc_provider=self._doc_provider,
            source_provider=self._source_provider,
            config=self._config,
            ticket_creator=self._ticket_creator,
        )
