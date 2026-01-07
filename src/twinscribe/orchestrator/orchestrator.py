"""
Dual-Stream Orchestrator.

Main orchestrator that coordinates the entire documentation pipeline:
- Initialization of static analysis and components
- Parallel stream execution
- Comparison and convergence
- Beads integration for discrepancy resolution
- Final output generation
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class OrchestratorPhase(str, Enum):
    """Current phase of the orchestrator."""

    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    DOCUMENTING = "documenting"
    COMPARING = "comparing"
    RESOLVING = "resolving"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchestratorConfig(BaseModel):
    """Configuration for the DualStreamOrchestrator.

    Attributes:
        max_iterations: Maximum documentation iterations
        parallel_components: Number of components to process in parallel
        wait_for_beads: Whether to wait for Beads resolution
        beads_timeout_hours: Timeout for Beads ticket resolution
        skip_validation: Skip validation step (for testing)
        dry_run: Don't create Beads tickets or write output
        continue_on_error: Continue processing if individual components fail
    """

    max_iterations: int = Field(default=5, ge=1, le=20)
    parallel_components: int = Field(default=10, ge=1, le=100)
    wait_for_beads: bool = Field(default=True)
    beads_timeout_hours: int = Field(default=48, ge=0)
    skip_validation: bool = Field(default=False)
    dry_run: bool = Field(default=False)
    continue_on_error: bool = Field(default=True)


@dataclass
class OrchestratorState:
    """Current state of the orchestrator.

    Attributes:
        phase: Current execution phase
        iteration: Current iteration number
        total_components: Total components to document
        processed_components: Components processed so far
        converged_components: Components that have converged
        pending_discrepancies: Discrepancies awaiting resolution
        beads_tickets_open: Number of open Beads tickets
        start_time: When processing started
        errors: List of errors encountered
    """

    phase: OrchestratorPhase = OrchestratorPhase.NOT_STARTED
    iteration: int = 0
    total_components: int = 0
    processed_components: int = 0
    converged_components: int = 0
    pending_discrepancies: int = 0
    beads_tickets_open: int = 0
    start_time: datetime | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "phase": self.phase.value,
            "iteration": self.iteration,
            "total_components": self.total_components,
            "processed_components": self.processed_components,
            "converged_components": self.converged_components,
            "pending_discrepancies": self.pending_discrepancies,
            "beads_tickets_open": self.beads_tickets_open,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "errors": self.errors,
        }


@dataclass
class IterationResult:
    """Result of a single iteration.

    Attributes:
        iteration: Iteration number
        components_processed: Number of components processed
        discrepancies_found: Number of discrepancies found
        discrepancies_resolved: Discrepancies resolved by ground truth
        beads_tickets_created: Beads tickets created
        converged: Whether convergence was achieved
        metrics: Additional metrics
    """

    iteration: int
    components_processed: int
    discrepancies_found: int
    discrepancies_resolved: int
    beads_tickets_created: int
    converged: bool
    metrics: dict[str, Any] = field(default_factory=dict)


# Type alias for progress callback
ProgressCallback = Callable[[OrchestratorState], None]


class DualStreamOrchestrator:
    """Main orchestrator for the dual-stream documentation system.

    Coordinates the full pipeline:
    1. Initialization - Parse codebase, run static analysis, build dependency graph
    2. Iteration loop - Document, compare, resolve discrepancies
    3. Finalization - Merge outputs, generate rebuild tickets

    Usage:
        orchestrator = DualStreamOrchestrator(
            config=config,
            static_oracle=oracle,
            stream_a=stream_a,
            stream_b=stream_b,
            comparator=comparator,
            beads_manager=beads_manager,
        )

        result = await orchestrator.run()
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        static_oracle: "StaticAnalysisOracle",
        stream_a: "DocumentationStream",
        stream_b: "DocumentationStream",
        comparator: "ComparatorAgent",
        beads_manager: Optional["BeadsLifecycleManager"] = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            static_oracle: Static analysis oracle for ground truth
            stream_a: First documentation stream
            stream_b: Second documentation stream
            comparator: Comparator agent for comparing outputs
            beads_manager: Optional Beads manager for ticket handling
        """
        self._config = config
        self._oracle = static_oracle
        self._stream_a = stream_a
        self._stream_b = stream_b
        self._comparator = comparator
        self._beads_manager = beads_manager

        # State
        self._state = OrchestratorState()
        self._iteration_history: list[IterationResult] = []
        self._progress_callbacks: list[ProgressCallback] = []

        # Component tracking
        self._components: list[Component] = []
        self._processing_order: list[str] = []
        self._component_results: dict[str, ComponentFinalDoc] = {}
        self._source_code_map: dict[str, str] = {}  # component_id -> source code

    @property
    def config(self) -> OrchestratorConfig:
        """Get orchestrator configuration."""
        return self._config

    @property
    def state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self._state

    @property
    def iteration_history(self) -> list[IterationResult]:
        """Get iteration history."""
        return self._iteration_history

    def on_progress(self, callback: ProgressCallback) -> None:
        """Register a progress callback.

        Args:
            callback: Function called with state updates
        """
        self._progress_callbacks.append(callback)

    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self._state)
            except Exception:
                pass  # Don't let callback errors affect processing

    async def run(self) -> "DocumentationPackage":
        """Execute the full documentation pipeline.

        Returns:
            Final documentation package

        Raises:
            OrchestratorError: If pipeline fails
        """
        try:
            self._state.start_time = datetime.utcnow()
            self._state.phase = OrchestratorPhase.INITIALIZING
            self._notify_progress()

            # Phase 1: Initialization
            await self._initialize()

            # Phase 2: Iteration loop
            converged = await self._run_iteration_loop()

            # Phase 3: Finalization
            self._state.phase = OrchestratorPhase.FINALIZING
            self._notify_progress()

            result = await self._finalize(converged)

            self._state.phase = OrchestratorPhase.COMPLETED
            self._notify_progress()

            return result

        except Exception as e:
            self._state.phase = OrchestratorPhase.FAILED
            self._state.errors.append(str(e))
            self._notify_progress()
            raise OrchestratorError(f"Pipeline failed: {e}") from e

    async def _initialize(self) -> None:
        """Initialize the pipeline.

        Steps:
        1. Ensure static analysis is ready
        2. Discover components
        3. Build processing order
        4. Initialize streams
        """
        # Ensure oracle is ready
        if not self._oracle.is_initialized:
            await self._oracle.initialize()

        # Discover components
        self._state.phase = OrchestratorPhase.DISCOVERING
        self._notify_progress()

        self._components = await self._discover_components()
        self._state.total_components = len(self._components)

        # Build processing order (topological sort)
        self._processing_order = self._build_processing_order()

        # Initialize streams if needed
        await self._stream_a.initialize()
        await self._stream_b.initialize()

        # Initialize Beads manager if available
        if self._beads_manager:
            await self._beads_manager.initialize()

    async def _discover_components(self) -> list["Component"]:
        """Discover all documentable components.

        Uses AST parsing to find all functions, methods, and classes
        in the codebase. The call graph from the oracle is used to
        compute processing order (dependencies before dependents).

        Returns:
            List of components to document
        """
        from twinscribe.analysis.component_discovery import ComponentDiscovery

        # Get the codebase path from the oracle
        codebase_path = self._oracle.codebase_path

        # Create component discovery with default settings
        discovery = ComponentDiscovery(
            codebase_path=codebase_path,
            include_private=False,  # Only public components by default
        )

        # Get the call graph from the oracle for topological ordering
        call_graph = self._oracle.call_graph

        # Discover components
        result = await discovery.discover(call_graph=call_graph)

        # Store source code map for later use by streams
        self._source_code_map = result.source_code_map

        # Store processing order
        self._processing_order = result.processing_order

        # Log discovery results
        if result.errors:
            for error in result.errors:
                self._state.errors.append(error)

        return result.components

    def _build_processing_order(self) -> list[str]:
        """Build topological processing order.

        Dependencies are processed before dependents.

        Returns:
            List of component IDs in processing order
        """
        # Get call graph from oracle
        call_graph = self._oracle.call_graph

        # Build dependency graph (reverse of call graph for processing order)
        # A depends on B if A calls B, so B should be processed first
        in_degree: dict[str, int] = {}
        adjacency: dict[str, list[str]] = {}

        component_ids = {c.component_id for c in self._components}

        for comp_id in component_ids:
            in_degree[comp_id] = 0
            adjacency[comp_id] = []

        # Build graph edges
        for edge in call_graph.edges:
            if edge.caller in component_ids and edge.callee in component_ids:
                # Caller depends on callee, so callee -> caller
                adjacency[edge.callee].append(edge.caller)
                in_degree[edge.caller] += 1

        # Kahn's algorithm for topological sort
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle cycles (components not in result)
        remaining = [cid for cid in component_ids if cid not in result]
        result.extend(sorted(remaining))

        return result

    async def _run_iteration_loop(self) -> bool:
        """Run the main iteration loop.

        Returns:
            True if converged, False if max iterations reached
        """
        while self._state.iteration < self._config.max_iterations:
            self._state.iteration += 1
            self._state.phase = OrchestratorPhase.DOCUMENTING
            self._notify_progress()

            # Run single iteration
            result = await self._run_iteration()
            self._iteration_history.append(result)

            # Update state
            self._state.converged_components = result.components_processed
            self._state.pending_discrepancies = (
                result.discrepancies_found - result.discrepancies_resolved
            )
            self._state.beads_tickets_open = result.beads_tickets_created

            if result.converged:
                return True

        return False

    async def _run_iteration(self) -> IterationResult:
        """Run a single iteration.

        Steps:
        1. Parallel documentation (both streams)
        2. Comparison
        3. Resolution (ground truth + Beads)
        4. Check convergence

        Returns:
            Iteration result
        """
        iteration = self._state.iteration

        # Step 1: Parallel documentation
        components_to_process = self._get_components_to_process()
        self._state.processed_components = 0

        output_a, output_b = await asyncio.gather(
            self._stream_a.process(components_to_process, self._processing_order),
            self._stream_b.process(components_to_process, self._processing_order),
        )

        self._state.processed_components = len(components_to_process)
        self._notify_progress()

        # Step 2: Comparison
        self._state.phase = OrchestratorPhase.COMPARING
        self._notify_progress()

        comparison = await self._comparator.compare(
            output_a,
            output_b,
            self._oracle.call_graph,
        )

        # Step 3: Resolution
        self._state.phase = OrchestratorPhase.RESOLVING
        self._notify_progress()

        resolved_by_ground_truth = 0
        beads_tickets_created = 0

        for discrepancy in comparison.discrepancies:
            if discrepancy.is_call_graph_related and discrepancy.ground_truth is not None:
                # Apply ground truth resolution
                await self._apply_ground_truth_resolution(discrepancy)
                resolved_by_ground_truth += 1

            elif discrepancy.requires_beads and self._beads_manager:
                # Create Beads ticket
                if not self._config.dry_run:
                    await self._create_beads_ticket(discrepancy)
                    beads_tickets_created += 1

        # Wait for Beads resolution if configured
        if beads_tickets_created > 0 and self._config.wait_for_beads and self._beads_manager:
            await self._wait_for_beads_resolution()

        # Step 4: Check convergence
        converged = self._check_convergence(comparison)

        return IterationResult(
            iteration=iteration,
            components_processed=len(components_to_process),
            discrepancies_found=len(comparison.discrepancies),
            discrepancies_resolved=resolved_by_ground_truth,
            beads_tickets_created=beads_tickets_created,
            converged=converged,
            metrics={
                "identical_components": comparison.summary.identical,
                "call_graph_match_rate": comparison.summary.agreement_rate,
                "documentation_similarity": comparison.summary.agreement_rate,
            },
        )

    def _get_components_to_process(self) -> list["Component"]:
        """Get components to process in this iteration.

        First iteration: all components
        Subsequent iterations: only components with discrepancies

        Returns:
            List of components to process
        """
        if self._state.iteration == 1:
            return self._components

        # Get components with unresolved discrepancies
        # This would check the comparison results from previous iteration
        # For now, return all components (optimization for later)
        return self._components

    async def _apply_ground_truth_resolution(
        self,
        discrepancy: "Discrepancy",
    ) -> None:
        """Apply a ground truth resolution to streams.

        Args:
            discrepancy: Discrepancy to resolve
        """
        from twinscribe.models.base import ResolutionAction

        # Determine field from discrepancy type
        field = discrepancy.type.value

        if discrepancy.resolution == ResolutionAction.ACCEPT_STREAM_A:
            await self._stream_b.apply_correction(
                discrepancy.component_id,
                field,
                discrepancy.stream_a_value,
            )
        elif discrepancy.resolution == ResolutionAction.ACCEPT_STREAM_B:
            await self._stream_a.apply_correction(
                discrepancy.component_id,
                field,
                discrepancy.stream_b_value,
            )
        elif discrepancy.resolution == ResolutionAction.ACCEPT_GROUND_TRUTH:
            # Apply ground truth to both streams
            await self._stream_a.apply_correction(
                discrepancy.component_id,
                field,
                discrepancy.ground_truth,
            )
            await self._stream_b.apply_correction(
                discrepancy.component_id,
                field,
                discrepancy.ground_truth,
            )

    async def _create_beads_ticket(self, discrepancy: "Discrepancy") -> str:
        """Create a Beads ticket for a discrepancy.

        Args:
            discrepancy: Discrepancy requiring human review

        Returns:
            Ticket key
        """
        from twinscribe.beads import DiscrepancyTemplateData

        # Look up component info for additional context
        component = next(
            (c for c in self._components if c.component_id == discrepancy.component_id),
            None,
        )
        component_type = component.type.value if component else "unknown"
        file_path = component.location.file_path if component else ""

        data = DiscrepancyTemplateData(
            discrepancy_id=discrepancy.discrepancy_id,
            component_name=discrepancy.component_id,
            component_type=component_type,
            file_path=file_path,
            discrepancy_type=discrepancy.type.value,
            stream_a_value=str(discrepancy.stream_a_value),
            stream_b_value=str(discrepancy.stream_b_value),
            static_analysis_value=(
                str(discrepancy.ground_truth) if discrepancy.ground_truth is not None else None
            ),
            context="",
            iteration=self._state.iteration,
        )

        tracked = await self._beads_manager.create_discrepancy_ticket(data)
        return tracked.ticket_key

    async def _wait_for_beads_resolution(self) -> None:
        """Wait for Beads tickets to be resolved."""
        timeout_seconds = self._config.beads_timeout_hours * 3600

        pending = self._beads_manager.tracker.get_pending_tickets()
        for ticket in pending:
            resolution = await self._beads_manager.wait_for_resolution(
                ticket.ticket_key,
                timeout_seconds=timeout_seconds,
            )

            if resolution:
                # Apply resolution
                await self._apply_beads_resolution(resolution)

    async def _apply_beads_resolution(
        self,
        resolution: "TicketResolution",
    ) -> None:
        """Apply a Beads resolution.

        Args:
            resolution: Resolution from human reviewer
        """
        from twinscribe.beads import ResolutionAction

        # Get the discrepancy
        tracked = self._beads_manager.tracker.get(resolution.ticket_key)
        if not tracked:
            return

        component_id = tracked.component_id
        if not component_id:
            return

        # Apply based on action
        if resolution.action == ResolutionAction.ACCEPT_A:
            # B should adopt A's value (get from stream A)
            pass  # Would need to fetch A's value
        elif resolution.action == ResolutionAction.ACCEPT_B:
            # A should adopt B's value (get from stream B)
            pass  # Would need to fetch B's value
        elif resolution.action in (ResolutionAction.MERGE, ResolutionAction.MANUAL):
            # Both adopt the provided content
            if resolution.content:
                await self._stream_a.apply_correction(
                    component_id,
                    tracked.metadata.get("field", "documentation"),
                    resolution.content,
                )
                await self._stream_b.apply_correction(
                    component_id,
                    tracked.metadata.get("field", "documentation"),
                    resolution.content,
                )

    def _check_convergence(self, comparison: "ComparisonResult") -> bool:
        """Check if convergence criteria are met.

        Args:
            comparison: Latest comparison result

        Returns:
            True if converged
        """
        # Check convergence status from comparator
        if comparison.convergence_status.converged:
            return True

        # Additional checks
        summary = comparison.summary

        # Check agreement rate
        if summary.agreement_rate < 0.95:
            return False

        # Check blocking discrepancies
        blocking = [d for d in comparison.discrepancies if d.is_blocking]
        if len(blocking) > 0:
            return False

        return True

    async def _finalize(self, converged: bool) -> "DocumentationPackage":
        """Finalize the documentation.

        Args:
            converged: Whether streams converged

        Returns:
            Final documentation package
        """
        from twinscribe.models import DocumentationPackage

        # Merge outputs from both streams
        final_docs = await self._merge_outputs()

        # Extract final call graph
        final_call_graph = self._extract_final_call_graph(final_docs)

        # Generate rebuild tickets if Beads is available
        rebuild_tickets = []
        if self._beads_manager and not self._config.dry_run:
            rebuild_tickets = await self._generate_rebuild_tickets(final_docs)

        # Generate convergence report
        convergence_report = self._generate_convergence_report(converged)

        # Convert list to dict for DocumentationPackage
        docs_dict = {doc.component_id: doc for doc in final_docs}

        # Convert rebuild_tickets to list of dicts
        rebuild_ticket_dicts = [{"ticket_key": t} for t in rebuild_tickets]

        return DocumentationPackage(
            documentation=docs_dict,
            call_graph=final_call_graph,
            rebuild_tickets=rebuild_ticket_dicts,
            convergence_report=convergence_report,
            metrics=self._calculate_final_metrics(),
        )

    async def _merge_outputs(self) -> list["ComponentFinalDoc"]:
        """Merge outputs from both streams.

        Returns:
            List of final component documentation
        """
        from twinscribe.models import ComponentFinalDoc

        # Get latest outputs from both streams
        output_a = self._stream_a.get_outputs()
        output_b = self._stream_b.get_outputs()

        merged = []
        for component in self._components:
            comp_id = component.component_id

            doc_a = output_a.get(comp_id)
            doc_b = output_b.get(comp_id)

            # Merge logic: prefer converged values, else use comparison result
            # For now, simple merge preferring A when identical
            if doc_a and doc_b:
                # Use A as base, they should be converged
                conf_a = doc_a.metadata.confidence if doc_a.metadata else 0.8
                conf_b = doc_b.metadata.confidence if doc_b.metadata else 0.8
                merged.append(
                    ComponentFinalDoc(
                        component_id=comp_id,
                        documentation=doc_a.documentation,
                        callers=[c.component_id for c in doc_a.call_graph.callers],
                        callees=[c.component_id for c in doc_a.call_graph.callees],
                        confidence_score=int(min(conf_a, conf_b) * 100),
                        source_stream="merged",
                    )
                )
            elif doc_a:
                conf_a = doc_a.metadata.confidence if doc_a.metadata else 0.8
                merged.append(
                    ComponentFinalDoc(
                        component_id=comp_id,
                        documentation=doc_a.documentation,
                        callers=[c.component_id for c in doc_a.call_graph.callers],
                        callees=[c.component_id for c in doc_a.call_graph.callees],
                        confidence_score=int(conf_a * 80),  # Lower confidence
                        source_stream="A",
                    )
                )
            elif doc_b:
                conf_b = doc_b.metadata.confidence if doc_b.metadata else 0.8
                merged.append(
                    ComponentFinalDoc(
                        component_id=comp_id,
                        documentation=doc_b.documentation,
                        callers=[c.component_id for c in doc_b.call_graph.callers],
                        callees=[c.component_id for c in doc_b.call_graph.callees],
                        confidence_score=int(conf_b * 80),
                        source_stream="B",
                    )
                )

        return merged

    def _extract_final_call_graph(
        self,
        docs: list["ComponentFinalDoc"],
    ) -> "CallGraph":
        """Extract final call graph from documentation.

        Args:
            docs: Final documentation

        Returns:
            Merged call graph
        """
        from twinscribe.models.base import CallType
        from twinscribe.models.call_graph import CallEdge, CallGraph

        edges = []
        for doc in docs:
            # Create edges from callers and callees
            for callee_id in doc.callees:
                edges.append(
                    CallEdge(
                        caller=doc.component_id,
                        callee=callee_id,
                        call_type=CallType.DIRECT,
                    )
                )

        return CallGraph(edges=edges)

    async def _generate_rebuild_tickets(
        self,
        docs: list["ComponentFinalDoc"],
    ) -> list[str]:
        """Generate Beads rebuild tickets.

        Args:
            docs: Final documentation

        Returns:
            List of ticket keys
        """
        from twinscribe.beads import RebuildTemplateData

        ticket_keys = []

        for priority, doc in enumerate(docs, start=1):
            # Look up component info
            component = next(
                (c for c in self._components if c.component_id == doc.component_id),
                None,
            )
            component_type = component.type.value if component else "component"
            file_path = component.location.file_path if component else ""

            data = RebuildTemplateData(
                component_name=doc.component_id,
                component_type=component_type,
                file_path=file_path,
                documentation=str(doc.documentation),
                call_graph={
                    "calls": doc.callees,
                    "called_by": doc.callers,
                },
                rebuild_priority=priority,
                complexity_score=0.5,
            )

            tracked = await self._beads_manager.create_rebuild_ticket(data)
            ticket_keys.append(tracked.ticket_key)

        return ticket_keys

    def _generate_convergence_report(self, converged: bool) -> "ConvergenceReport":
        """Generate convergence report.

        Args:
            converged: Whether convergence was achieved

        Returns:
            Convergence report
        """
        from twinscribe.models import ConvergenceHistoryEntry, ConvergenceReport

        history = []
        for result in self._iteration_history:
            history.append(
                ConvergenceHistoryEntry(
                    iteration=result.iteration,
                    components_processed=result.components_processed,
                    discrepancies_found=result.discrepancies_found,
                    discrepancies_resolved=result.discrepancies_resolved,
                    call_graph_match_rate=result.metrics.get("call_graph_match_rate", 0),
                    documentation_similarity=result.metrics.get("documentation_similarity", 0),
                )
            )

        return ConvergenceReport(
            converged=converged,
            total_iterations=self._state.iteration,
            final_status="converged" if converged else "max_iterations_reached",
            history=history,
        )

    def _calculate_final_metrics(self) -> "RunMetrics":
        """Calculate final metrics.

        Returns:
            RunMetrics object
        """
        from twinscribe.models.output import RunMetrics

        # Calculate totals from iteration history
        total_discrepancies = sum(r.discrepancies_found for r in self._iteration_history)
        auto_resolved = sum(r.discrepancies_resolved for r in self._iteration_history)
        beads_resolved = sum(r.beads_tickets_created for r in self._iteration_history)

        return RunMetrics(
            run_id=f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            codebase_path=str(self._oracle.codebase_path),
            started_at=self._state.start_time or datetime.utcnow(),
            completed_at=datetime.utcnow(),
            components_total=self._state.total_components,
            components_documented=self._state.processed_components,
            discrepancies_total=total_discrepancies,
            discrepancies_resolved_auto=auto_resolved,
            discrepancies_resolved_beads=beads_resolved,
            discrepancies_unresolved=self._state.pending_discrepancies,
        )


class OrchestratorError(Exception):
    """Raised when orchestrator encounters an error."""

    pass
