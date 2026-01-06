# Orchestrator Workflow and Iteration Loop Design Document

## Version 1.0 | January 2026

---

## 1. Overview

This document specifies the architecture for the DualStreamOrchestrator class, which coordinates the entire documentation pipeline including initialization, parallel stream processing, comparison, resolution, and finalization phases.

### Design Goals

1. **Coordination**: Manage lifecycle of all components (streams, comparator, static oracle)
2. **Parallelism**: Maximize throughput via parallel stream execution
3. **State Management**: Track iteration progress, convergence history, and errors
4. **Resilience**: Handle failures gracefully with recovery and partial results
5. **Observability**: Provide comprehensive logging and metrics

---

## 2. Class Architecture

```
                        +------------------------+
                        | DualStreamOrchestrator |
                        +------------+-----------+
                                     |
         +---------------------------+---------------------------+
         |              |            |            |              |
         v              v            v            v              v
+--------+------+ +-----+------+ +---+----+ +----+-----+ +-------+-------+
| Component     | | Dependency | | Static | | Stream   | | Beads         |
| Discovery     | | Graph      | | Oracle | | Manager  | | Lifecycle     |
| <<component>> | | Builder    | |        | |          | | Manager       |
+---------------+ +------------+ +--------+ +----------+ +---------------+
         |              |            |            |              |
         v              v            v            v              v
+--------+------+ +-----+------+ +---+----+ +----+-----+ +-------+-------+
| ASTParser     | | Topological| | PyCG   | | Stream A | | Ticket        |
| ComponentFinder| | Sorter    | | pyan3  | | Stream B | | Resolution    |
+---------------+ +------------+ +--------+ +----------+ +---------------+
```

---

## 3. Core Data Models

```python
# twinscribe/orchestrator/models.py

from datetime import datetime
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


class OrchestratorPhase(str, Enum):
    """Current phase of the orchestrator."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    PROCESSING = "processing"
    COMPARING = "comparing"
    RESOLVING = "resolving"
    WAITING_FOR_BEADS = "waiting_for_beads"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentInfo(BaseModel):
    """Information about a discovered component."""
    component_id: str
    component_type: str  # 'function', 'method', 'class', 'module'
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    is_public: bool = True
    complexity_score: float = 0.0


class ProcessingOrder(BaseModel):
    """Topologically sorted processing order."""
    components: list[str]  # Component IDs in order
    levels: dict[str, int]  # Component ID -> depth level
    dependency_count: dict[str, int]  # Component ID -> number of dependencies


class IterationState(BaseModel):
    """State for a single iteration."""
    iteration_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    stream_a_completed: bool = False
    stream_b_completed: bool = False
    comparison_completed: bool = False
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    beads_tickets_created: int = 0
    corrections_applied: int = 0
    convergence_status: Optional[dict] = None


class OrchestratorState(BaseModel):
    """Complete orchestrator state."""
    run_id: str
    phase: OrchestratorPhase = OrchestratorPhase.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Discovery state
    total_components: int = 0
    components_discovered: list[str] = Field(default_factory=list)
    processing_order: Optional[ProcessingOrder] = None

    # Iteration state
    current_iteration: int = 0
    max_iterations: int = 5
    iterations: list[IterationState] = Field(default_factory=list)

    # Convergence state
    converged: bool = False
    final_convergence_status: Optional[dict] = None

    # Error state
    errors: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)


class DocumentationPackage(BaseModel):
    """Final output package from the orchestrator."""
    documentation: dict[str, Any]  # component_id -> documentation
    call_graph: dict[str, Any]  # Complete call graph
    rebuild_tickets: list[dict]  # Beads rebuild tickets
    convergence_report: dict  # Convergence history and metrics
    run_metadata: dict  # Run statistics and timing
    warnings: list[str] = Field(default_factory=list)  # Non-blocking issues


class OrchestratorConfig(BaseModel):
    """Configuration for the orchestrator."""
    codebase_path: str
    language: str = "python"
    max_iterations: int = 5
    parallel_streams: bool = True
    beads_timeout_hours: int = 48
    force_converge_on_timeout: bool = True
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "**/test_*",
        "**/tests/**",
        "**/__pycache__/**",
        "**/.git/**",
    ])
    include_private: bool = False
    min_complexity_score: float = 0.0
```

---

## 4. DualStreamOrchestrator Implementation

```python
# twinscribe/orchestrator/orchestrator.py

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Any
import logging

from twinscribe.orchestrator.models import (
    OrchestratorPhase,
    OrchestratorState,
    OrchestratorConfig,
    ComponentInfo,
    ProcessingOrder,
    IterationState,
    DocumentationPackage,
)
from twinscribe.orchestrator.discovery import ComponentDiscovery
from twinscribe.orchestrator.dependency import DependencyGraphBuilder
from twinscribe.streams.documentation_stream import DocumentationStream
from twinscribe.agents.comparator.agent import ComparatorAgent
from twinscribe.agents.comparator.models import ComparisonResult
from twinscribe.analysis.static_oracle import StaticAnalysisOracle
from twinscribe.beads.lifecycle import BeadsLifecycleManager
from twinscribe.config.models import AgentSystemConfig


logger = logging.getLogger(__name__)


class DualStreamOrchestrator:
    """
    Main orchestrator for the dual-stream documentation system.

    Responsibilities:
    - Initialize and configure all system components
    - Discover and analyze codebase components
    - Coordinate parallel stream execution
    - Manage iteration loop until convergence
    - Handle Beads ticket lifecycle
    - Produce final documentation package

    Lifecycle:
    1. __init__: Configure components
    2. run(): Execute full pipeline
       - _initialize(): Setup phase
       - _iteration_loop(): Main processing
       - _finalize(): Output generation
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        agent_config: AgentSystemConfig,
        beads_config: dict,
        llm_client: "LLMClient",
    ):
        self.config = config
        self.agent_config = agent_config
        self.llm_client = llm_client

        # Generate unique run ID
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Initialize state
        self.state = OrchestratorState(
            run_id=self.run_id,
            max_iterations=config.max_iterations,
        )

        # Components (initialized in _initialize)
        self.static_oracle: Optional[StaticAnalysisOracle] = None
        self.stream_a: Optional[DocumentationStream] = None
        self.stream_b: Optional[DocumentationStream] = None
        self.comparator: Optional[ComparatorAgent] = None
        self.beads_manager: Optional[BeadsLifecycleManager] = None

        # Discovery components
        self.component_discovery = ComponentDiscovery(config.codebase_path)
        self.dependency_builder = DependencyGraphBuilder()

        # Internal state
        self._components: dict[str, ComponentInfo] = {}
        self._processing_order: list[str] = []
        self._comparison_history: list[ComparisonResult] = []

    async def run(self) -> DocumentationPackage:
        """
        Execute the full documentation pipeline.

        Returns:
            DocumentationPackage with all outputs

        Raises:
            OrchestratorError: If pipeline fails unrecoverably
        """
        try:
            self.state.started_at = datetime.now()
            self.state.phase = OrchestratorPhase.INITIALIZING

            logger.info(f"Starting documentation run: {self.run_id}")

            # Phase 1: Initialize
            await self._initialize()

            # Phase 2: Iteration loop
            await self._iteration_loop()

            # Phase 3: Finalize
            result = await self._finalize()

            self.state.completed_at = datetime.now()
            self.state.phase = OrchestratorPhase.COMPLETED

            logger.info(f"Documentation run completed: {self.run_id}")
            return result

        except Exception as e:
            self.state.phase = OrchestratorPhase.FAILED
            self.state.errors.append({
                "phase": "run",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            logger.error(f"Documentation run failed: {e}")
            raise OrchestratorError(f"Pipeline failed: {e}") from e

    # =========================================================================
    # PHASE 1: INITIALIZATION
    # =========================================================================

    async def _initialize(self) -> None:
        """
        Initialize all components and discover codebase structure.

        Steps:
        1. Parse codebase to AST
        2. Discover all documentable components
        3. Build dependency graph
        4. Run static analysis for ground truth
        5. Compute topological processing order
        6. Initialize both streams
        """
        logger.info("Initializing orchestrator...")

        # Step 1: Discover components
        self.state.phase = OrchestratorPhase.DISCOVERING
        self._components = await self._discover_components()
        self.state.total_components = len(self._components)
        self.state.components_discovered = list(self._components.keys())

        logger.info(f"Discovered {len(self._components)} components")

        # Step 2: Build dependency graph and get processing order
        dependency_graph = self.dependency_builder.build(self._components)
        self._processing_order = self._topological_sort(dependency_graph)

        self.state.processing_order = ProcessingOrder(
            components=self._processing_order,
            levels=self._compute_levels(dependency_graph),
            dependency_count={c: len(self._components[c].dependencies) for c in self._components},
        )

        logger.info(f"Processing order computed: {len(self._processing_order)} components")

        # Step 3: Initialize static analysis oracle
        self.static_oracle = StaticAnalysisOracle(
            self.config.codebase_path,
            self.config.language,
        )
        await self.static_oracle.analyze()

        logger.info(f"Static analysis complete: {len(self.static_oracle.call_graph.edges)} edges")

        # Step 4: Initialize streams
        self.stream_a = self._create_stream("A", self.agent_config.stream_a)
        self.stream_b = self._create_stream("B", self.agent_config.stream_b)

        # Step 5: Initialize comparator
        self.comparator = ComparatorAgent(
            agent_id="C",
            llm_client=self.llm_client,
            static_oracle=self.static_oracle,
        )

        # Step 6: Initialize Beads manager
        self.beads_manager = BeadsLifecycleManager(self.beads_config)

        logger.info("Initialization complete")

    async def _discover_components(self) -> dict[str, ComponentInfo]:
        """Discover all documentable components in the codebase."""
        raw_components = await self.component_discovery.discover(
            exclude_patterns=self.config.exclude_patterns,
            include_private=self.config.include_private,
        )

        # Filter by complexity if configured
        if self.config.min_complexity_score > 0:
            raw_components = {
                cid: comp for cid, comp in raw_components.items()
                if comp.complexity_score >= self.config.min_complexity_score
            }

        return raw_components

    def _topological_sort(self, dependency_graph: dict[str, set[str]]) -> list[str]:
        """
        Perform topological sort to determine processing order.

        Components are sorted so that dependencies are processed before dependents.
        This ensures documentation of dependencies is available as context.
        """
        # Kahn's algorithm
        in_degree = {node: 0 for node in dependency_graph}
        for node in dependency_graph:
            for dep in dependency_graph[node]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self._get_dependents(node, dependency_graph):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Handle cycles by adding remaining nodes
        remaining = set(dependency_graph.keys()) - set(result)
        if remaining:
            logger.warning(f"Dependency cycle detected, adding {len(remaining)} components")
            result.extend(remaining)

        return result

    def _get_dependents(
        self,
        node: str,
        dependency_graph: dict[str, set[str]]
    ) -> list[str]:
        """Get all nodes that depend on the given node."""
        dependents = []
        for candidate, deps in dependency_graph.items():
            if node in deps:
                dependents.append(candidate)
        return dependents

    def _compute_levels(self, dependency_graph: dict[str, set[str]]) -> dict[str, int]:
        """Compute depth level for each component."""
        levels = {}

        def get_level(node: str, visited: set) -> int:
            if node in levels:
                return levels[node]
            if node in visited:
                return 0  # Cycle

            visited.add(node)
            deps = dependency_graph.get(node, set())
            if not deps:
                levels[node] = 0
            else:
                levels[node] = 1 + max(
                    get_level(dep, visited) for dep in deps if dep in dependency_graph
                )
            return levels[node]

        for node in dependency_graph:
            get_level(node, set())

        return levels

    def _create_stream(
        self,
        stream_id: str,
        stream_config: "StreamConfig"
    ) -> DocumentationStream:
        """Create and configure a documentation stream."""
        from twinscribe.agents.documenter.agent import (
            ClaudeDocumenterAgent,
            OpenAIDocumenterAgent,
        )
        from twinscribe.agents.validator.agent import (
            ClaudeValidatorAgent,
            OpenAIValidatorAgent,
        )

        # Create documenter based on provider
        if stream_config.documenter_model.provider == "anthropic":
            documenter = ClaudeDocumenterAgent(
                agent_id=f"{stream_id}1",
                llm_client=self.llm_client,
            )
        else:
            documenter = OpenAIDocumenterAgent(
                agent_id=f"{stream_id}1",
                llm_client=self.llm_client,
            )

        # Create validator based on provider
        if stream_config.validator_model.provider == "anthropic":
            validator = ClaudeValidatorAgent(
                agent_id=f"{stream_id}2",
                llm_client=self.llm_client,
            )
        else:
            validator = OpenAIValidatorAgent(
                agent_id=f"{stream_id}2",
                llm_client=self.llm_client,
            )

        return DocumentationStream(
            stream_id=stream_id,
            documenter=documenter,
            validator=validator,
            static_oracle=self.static_oracle,
        )

    # =========================================================================
    # PHASE 2: ITERATION LOOP
    # =========================================================================

    async def _iteration_loop(self) -> None:
        """
        Main iteration loop until convergence or max iterations.

        Each iteration:
        1. Run parallel documentation streams
        2. Compare outputs
        3. Check convergence
        4. Handle discrepancies (corrections or Beads tickets)
        5. Update state
        """
        self.state.phase = OrchestratorPhase.PROCESSING

        while self.state.current_iteration < self.state.max_iterations:
            self.state.current_iteration += 1
            iteration_num = self.state.current_iteration

            logger.info(f"=== Iteration {iteration_num} ===")

            # Create iteration state
            iteration_state = IterationState(
                iteration_number=iteration_num,
                started_at=datetime.now(),
            )

            # Step 1: Parallel documentation
            logger.info("Running parallel documentation streams...")
            stream_a_results, stream_b_results = await self._run_parallel_streams(
                iteration_num
            )
            iteration_state.stream_a_completed = True
            iteration_state.stream_b_completed = True

            # Step 2: Compare outputs
            logger.info("Comparing outputs...")
            self.state.phase = OrchestratorPhase.COMPARING
            comparison = await self._compare_outputs(
                stream_a_results,
                stream_b_results,
                iteration_num,
            )
            iteration_state.comparison_completed = True
            iteration_state.discrepancies_found = comparison.summary.discrepancies

            self._comparison_history.append(comparison)

            # Log comparison summary
            logger.info(f"  Identical: {comparison.summary.identical}")
            logger.info(f"  Discrepancies: {comparison.summary.discrepancies}")
            logger.info(f"  Resolved by ground truth: {comparison.summary.resolved_by_ground_truth}")
            logger.info(f"  Requires human review: {comparison.summary.requires_human_review}")

            # Step 3: Check convergence
            if comparison.convergence_status.converged:
                logger.info("Streams converged!")
                self.state.converged = True
                self.state.final_convergence_status = comparison.convergence_status.model_dump()
                iteration_state.convergence_status = comparison.convergence_status.model_dump()
                iteration_state.completed_at = datetime.now()
                self.state.iterations.append(iteration_state)
                break

            # Step 4: Handle discrepancies
            self.state.phase = OrchestratorPhase.RESOLVING
            await self._handle_discrepancies(comparison, iteration_state)

            # Step 5: Update state
            iteration_state.completed_at = datetime.now()
            iteration_state.convergence_status = comparison.convergence_status.model_dump()
            self.state.iterations.append(iteration_state)

            # Check if we should continue
            if comparison.convergence_status.recommendation == "force_converge":
                logger.warning("Max iterations reached, forcing convergence")
                self.state.warnings.append({
                    "type": "forced_convergence",
                    "iteration": iteration_num,
                    "open_discrepancies": comparison.convergence_status.open_discrepancies,
                })
                break

        # Set final convergence status if not already set
        if not self.state.final_convergence_status and self._comparison_history:
            self.state.final_convergence_status = (
                self._comparison_history[-1].convergence_status.model_dump()
            )

    async def _run_parallel_streams(
        self,
        iteration: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Run both documentation streams in parallel.

        Uses asyncio.gather for concurrent execution, maximizing throughput.
        Each stream processes components in topological order.
        """
        # Prepare component list for processing
        components = [
            self._components[cid].model_dump()
            for cid in self._processing_order
        ]

        if self.config.parallel_streams:
            # Run streams in parallel
            stream_a_task = self.stream_a.process(components, iteration)
            stream_b_task = self.stream_b.process(components, iteration)

            results_a, results_b = await asyncio.gather(
                stream_a_task,
                stream_b_task,
                return_exceptions=True,
            )

            # Handle exceptions
            if isinstance(results_a, Exception):
                logger.error(f"Stream A failed: {results_a}")
                raise OrchestratorError(f"Stream A failed: {results_a}")
            if isinstance(results_b, Exception):
                logger.error(f"Stream B failed: {results_b}")
                raise OrchestratorError(f"Stream B failed: {results_b}")

            return (
                {cid: r.documentation.model_dump() for cid, r in results_a.items()},
                {cid: r.documentation.model_dump() for cid, r in results_b.items()},
            )
        else:
            # Run sequentially (for debugging)
            results_a = await self.stream_a.process(components, iteration)
            results_b = await self.stream_b.process(components, iteration)

            return (
                {cid: r.documentation.model_dump() for cid, r in results_a.items()},
                {cid: r.documentation.model_dump() for cid, r in results_b.items()},
            )

    async def _compare_outputs(
        self,
        stream_a_output: dict[str, Any],
        stream_b_output: dict[str, Any],
        iteration: int,
    ) -> ComparisonResult:
        """Compare outputs from both streams."""
        previous = self._comparison_history[-1] if self._comparison_history else None

        return await self.comparator.compare(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            iteration=iteration,
            previous_comparison=previous,
        )

    async def _handle_discrepancies(
        self,
        comparison: ComparisonResult,
        iteration_state: IterationState
    ) -> None:
        """
        Handle discrepancies from comparison.

        Actions:
        1. Apply auto-resolved corrections to streams
        2. Create Beads tickets for human review
        3. Wait for ticket resolution if needed
        """
        # Apply corrections that were auto-resolved
        corrections_applied = 0

        for component_id, corrections in comparison.corrections_for_stream_a.items():
            await self.stream_a.apply_correction(component_id, corrections)
            corrections_applied += 1

        for component_id, corrections in comparison.corrections_for_stream_b.items():
            await self.stream_b.apply_correction(component_id, corrections)
            corrections_applied += 1

        iteration_state.corrections_applied = corrections_applied
        iteration_state.discrepancies_resolved = (
            comparison.summary.resolved_by_ground_truth +
            comparison.summary.resolved_by_judgment
        )

        # Handle Beads tickets if needed
        if comparison.summary.requires_human_review > 0:
            await self._handle_beads_tickets(comparison, iteration_state)

    async def _handle_beads_tickets(
        self,
        comparison: ComparisonResult,
        iteration_state: IterationState
    ) -> None:
        """Create and manage Beads tickets for human review."""
        self.state.phase = OrchestratorPhase.WAITING_FOR_BEADS

        ticket_keys = []

        for disc in comparison.discrepancies:
            if disc.requires_beads and disc.beads_ticket:
                try:
                    key = await self.beads_manager.create_discrepancy_ticket(disc)
                    ticket_keys.append(key)
                    logger.info(f"Created Beads ticket: {key}")
                except Exception as e:
                    logger.error(f"Failed to create Beads ticket: {e}")
                    self.state.warnings.append({
                        "type": "beads_ticket_failed",
                        "component_id": disc.component_id,
                        "error": str(e),
                    })

        iteration_state.beads_tickets_created = len(ticket_keys)

        if ticket_keys:
            logger.info(f"Waiting for {len(ticket_keys)} Beads tickets to be resolved...")

            resolution_result = await self.beads_manager.wait_for_resolution(
                ticket_keys,
                timeout_hours=self.config.beads_timeout_hours,
            )

            if resolution_result.all_resolved:
                logger.info("All Beads tickets resolved")
                for key in ticket_keys:
                    await self.beads_manager.apply_resolution(key)
            else:
                logger.warning(f"Timeout: {len(resolution_result.pending)} tickets still open")
                if self.config.force_converge_on_timeout:
                    self.state.warnings.append({
                        "type": "beads_timeout",
                        "pending_tickets": resolution_result.pending,
                    })

    # =========================================================================
    # PHASE 3: FINALIZATION
    # =========================================================================

    async def _finalize(self) -> DocumentationPackage:
        """
        Finalize and produce the documentation package.

        Steps:
        1. Merge converged outputs
        2. Generate final documentation
        3. Create rebuild tickets
        4. Produce convergence report
        """
        self.state.phase = OrchestratorPhase.FINALIZING
        logger.info("Finalizing documentation package...")

        # Get final outputs from streams
        stream_a_outputs = self.stream_a.get_all_outputs()
        stream_b_outputs = self.stream_b.get_all_outputs()

        # Merge outputs based on comparison results
        final_documentation = await self._merge_outputs(
            stream_a_outputs,
            stream_b_outputs,
        )

        # Extract final call graph
        final_call_graph = self._extract_call_graph(final_documentation)

        # Generate rebuild tickets
        rebuild_tickets = await self._generate_rebuild_tickets(final_documentation)

        # Generate convergence report
        convergence_report = self._generate_convergence_report()

        # Compile run metadata
        run_metadata = self._compile_run_metadata()

        return DocumentationPackage(
            documentation=final_documentation,
            call_graph=final_call_graph,
            rebuild_tickets=rebuild_tickets,
            convergence_report=convergence_report,
            run_metadata=run_metadata,
            warnings=[w.get("type", "unknown") for w in self.state.warnings],
        )

    async def _merge_outputs(
        self,
        stream_a_outputs: dict[str, Any],
        stream_b_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge outputs from both streams into final documentation.

        Merge Strategy:
        - For identical outputs: use either (they're the same)
        - For resolved discrepancies: use the resolved value
        - For unresolved: use Stream A as default (with warning)
        """
        final = {}
        last_comparison = self._comparison_history[-1] if self._comparison_history else None

        # Build resolution map from last comparison
        resolutions = {}
        if last_comparison:
            for disc in last_comparison.discrepancies:
                resolutions[disc.component_id] = disc.resolution

        # Merge each component
        all_components = set(stream_a_outputs.keys()) | set(stream_b_outputs.keys())

        for component_id in all_components:
            a_doc = stream_a_outputs.get(component_id)
            b_doc = stream_b_outputs.get(component_id)

            if a_doc is None:
                final[component_id] = b_doc.model_dump() if b_doc else None
            elif b_doc is None:
                final[component_id] = a_doc.model_dump()
            else:
                # Both exist - check resolution
                resolution = resolutions.get(component_id)

                if resolution == ResolutionType.ACCEPT_STREAM_B:
                    final[component_id] = b_doc.model_dump()
                else:
                    # Default to Stream A
                    final[component_id] = a_doc.model_dump()

        return final

    def _extract_call_graph(self, documentation: dict[str, Any]) -> dict[str, Any]:
        """Extract complete call graph from documentation."""
        edges = []

        for component_id, doc in documentation.items():
            if doc is None:
                continue

            call_graph = doc.get("call_graph", {})

            # Add callee edges
            for callee in call_graph.get("callees", []):
                edges.append({
                    "caller": component_id,
                    "callee": callee.get("component_id"),
                    "call_site_line": callee.get("call_site_line"),
                    "call_type": callee.get("call_type"),
                })

        return {
            "edges": edges,
            "nodes": list(documentation.keys()),
            "total_edges": len(edges),
            "total_nodes": len(documentation),
        }

    async def _generate_rebuild_tickets(
        self,
        documentation: dict[str, Any]
    ) -> list[dict]:
        """Generate Beads rebuild tickets for each documented component."""
        tickets = []

        for component_id, doc in documentation.items():
            if doc is None:
                continue

            component_info = self._components.get(component_id)
            if component_info is None:
                continue

            ticket = self._build_rebuild_ticket(component_id, doc, component_info)
            tickets.append(ticket)

        return tickets

    def _build_rebuild_ticket(
        self,
        component_id: str,
        documentation: dict,
        component_info: ComponentInfo,
    ) -> dict:
        """Build a Beads rebuild ticket for a component."""
        doc_content = documentation.get("documentation", {})
        call_graph = documentation.get("call_graph", {})

        return {
            "project": "REBUILD",
            "issue_type": "Story",
            "priority": "Medium",
            "summary": f"Rebuild: {component_id}",
            "description": self._format_rebuild_description(
                component_id, doc_content, call_graph, component_info
            ),
            "labels": ["legacy-rebuild", "ai-documented"],
            "custom_fields": {
                "cf_component_id": component_id,
                "cf_file_path": component_info.file_path,
                "cf_callees_count": len(call_graph.get("callees", [])),
                "cf_callers_count": len(call_graph.get("callers", [])),
            },
        }

    def _format_rebuild_description(
        self,
        component_id: str,
        documentation: dict,
        call_graph: dict,
        component_info: ComponentInfo,
    ) -> str:
        """Format rebuild ticket description."""
        callees = call_graph.get("callees", [])
        callers = call_graph.get("callers", [])

        return f"""## Component Specification

**Current Location:** `{component_info.file_path}:{component_info.line_start}-{component_info.line_end}`
**Type:** {component_info.component_type}

## Purpose

{documentation.get('summary', 'No summary')}

## Detailed Description

{documentation.get('description', 'No description')}

## Interface Contract

### Parameters
{self._format_parameters(documentation.get('parameters', []))}

### Returns
{self._format_returns(documentation.get('returns'))}

### Exceptions
{self._format_exceptions(documentation.get('raises', []))}

## Call Graph

### This component calls ({len(callees)} dependencies):
{self._format_callees(callees)}

### Called by ({len(callers)} dependents):
{self._format_callers(callers)}

## Rebuild Checklist

- [ ] Implement documented interface exactly
- [ ] Preserve all {len(callees)} downstream dependencies
- [ ] Ensure compatibility with {len(callers)} upstream callers
- [ ] Add unit tests for documented exceptions
- [ ] Verify call graph matches specification
"""

    def _format_parameters(self, parameters: list) -> str:
        if not parameters:
            return "None"

        lines = ["| Name | Type | Description |", "|------|------|-------------|"]
        for param in parameters:
            lines.append(f"| `{param.get('name')}` | `{param.get('type', 'Any')}` | {param.get('description', '')} |")
        return "\n".join(lines)

    def _format_returns(self, returns: Optional[dict]) -> str:
        if not returns:
            return "None"
        return f"- **Type:** `{returns.get('type', 'None')}`\n- **Description:** {returns.get('description', '')}"

    def _format_exceptions(self, raises: list) -> str:
        if not raises:
            return "None"
        return "\n".join(f"- `{r.get('type')}`: {r.get('condition')}" for r in raises)

    def _format_callees(self, callees: list) -> str:
        if not callees:
            return "None"
        return "\n".join(
            f"- `{c.get('component_id')}` (line {c.get('call_site_line')}) - {c.get('call_type', 'direct')}"
            for c in callees
        )

    def _format_callers(self, callers: list) -> str:
        if not callers:
            return "None"
        return "\n".join(
            f"- `{c.get('component_id')}` (line {c.get('call_site_line')})"
            for c in callers
        )

    def _generate_convergence_report(self) -> dict:
        """Generate report on the convergence process."""
        return {
            "run_id": self.run_id,
            "total_iterations": self.state.current_iteration,
            "converged": self.state.converged,
            "final_status": self.state.final_convergence_status,
            "iteration_history": [
                {
                    "iteration": it.iteration_number,
                    "discrepancies_found": it.discrepancies_found,
                    "discrepancies_resolved": it.discrepancies_resolved,
                    "beads_tickets": it.beads_tickets_created,
                    "corrections": it.corrections_applied,
                    "duration_seconds": (
                        (it.completed_at - it.started_at).total_seconds()
                        if it.completed_at else None
                    ),
                }
                for it in self.state.iterations
            ],
            "comparison_summaries": [
                comp.summary.model_dump()
                for comp in self._comparison_history
            ],
        }

    def _compile_run_metadata(self) -> dict:
        """Compile metadata about the run."""
        total_duration = None
        if self.state.started_at and self.state.completed_at:
            total_duration = (self.state.completed_at - self.state.started_at).total_seconds()

        return {
            "run_id": self.run_id,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "completed_at": self.state.completed_at.isoformat() if self.state.completed_at else None,
            "duration_seconds": total_duration,
            "total_components": self.state.total_components,
            "total_iterations": self.state.current_iteration,
            "stream_a_stats": self.stream_a.get_statistics() if self.stream_a else None,
            "stream_b_stats": self.stream_b.get_statistics() if self.stream_b else None,
            "comparator_stats": self.comparator.get_statistics() if self.comparator else None,
            "errors": self.state.errors,
            "warnings": self.state.warnings,
        }

    # =========================================================================
    # PUBLIC STATE ACCESS
    # =========================================================================

    def get_state(self) -> OrchestratorState:
        """Get current orchestrator state."""
        return self.state

    def get_progress(self) -> dict:
        """Get current progress information."""
        return {
            "phase": self.state.phase.value,
            "iteration": self.state.current_iteration,
            "max_iterations": self.state.max_iterations,
            "components_total": self.state.total_components,
            "converged": self.state.converged,
        }
```

---

## 5. Component Discovery

```python
# twinscribe/orchestrator/discovery.py

import ast
from pathlib import Path
from typing import Optional
import fnmatch

from twinscribe.orchestrator.models import ComponentInfo


class ComponentDiscovery:
    """
    Discovers documentable components in a Python codebase.

    Discovers:
    - Functions (module-level)
    - Classes
    - Methods (instance, class, static)
    - Properties

    Uses AST parsing for accurate extraction.
    """

    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)

    async def discover(
        self,
        exclude_patterns: Optional[list[str]] = None,
        include_private: bool = False,
    ) -> dict[str, ComponentInfo]:
        """Discover all components in the codebase."""
        components = {}
        exclude_patterns = exclude_patterns or []

        for py_file in self.codebase_path.rglob("*.py"):
            # Check exclusions
            if self._should_exclude(py_file, exclude_patterns):
                continue

            try:
                file_components = self._parse_file(py_file, include_private)
                components.update(file_components)
            except SyntaxError as e:
                # Log but continue
                pass

        return components

    def _should_exclude(self, path: Path, patterns: list[str]) -> bool:
        """Check if path matches any exclusion pattern."""
        rel_path = str(path.relative_to(self.codebase_path))
        return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)

    def _parse_file(
        self,
        file_path: Path,
        include_private: bool
    ) -> dict[str, ComponentInfo]:
        """Parse a Python file and extract components."""
        components = {}
        source = file_path.read_text()
        tree = ast.parse(source)

        module_name = self._path_to_module(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                comp = self._extract_function(node, module_name, file_path, source)
                components[comp.component_id] = comp

            elif isinstance(node, ast.AsyncFunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                comp = self._extract_function(node, module_name, file_path, source)
                components[comp.component_id] = comp

            elif isinstance(node, ast.ClassDef):
                if not include_private and node.name.startswith("_"):
                    continue
                comp = self._extract_class(node, module_name, file_path, source)
                components[comp.component_id] = comp

                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not include_private and item.name.startswith("_"):
                            continue
                        method_comp = self._extract_method(
                            item, module_name, node.name, file_path, source
                        )
                        components[method_comp.component_id] = method_comp

        return components

    def _path_to_module(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.codebase_path)
        parts = list(rel_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace(".py", "")
        return ".".join(parts)

    def _extract_function(
        self,
        node: ast.FunctionDef,
        module: str,
        file_path: Path,
        source: str
    ) -> ComponentInfo:
        """Extract function component info."""
        return ComponentInfo(
            component_id=f"{module}.{node.name}",
            component_type="function",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=self._get_source_segment(source, node),
            dependencies=self._extract_dependencies(node),
            is_public=not node.name.startswith("_"),
            complexity_score=self._calculate_complexity(node),
        )

    def _extract_class(
        self,
        node: ast.ClassDef,
        module: str,
        file_path: Path,
        source: str
    ) -> ComponentInfo:
        """Extract class component info."""
        return ComponentInfo(
            component_id=f"{module}.{node.name}",
            component_type="class",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=self._get_source_segment(source, node),
            dependencies=self._extract_dependencies(node),
            is_public=not node.name.startswith("_"),
            complexity_score=self._calculate_complexity(node),
        )

    def _extract_method(
        self,
        node: ast.FunctionDef,
        module: str,
        class_name: str,
        file_path: Path,
        source: str
    ) -> ComponentInfo:
        """Extract method component info."""
        return ComponentInfo(
            component_id=f"{module}.{class_name}.{node.name}",
            component_type="method",
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            source_code=self._get_source_segment(source, node),
            dependencies=self._extract_dependencies(node),
            is_public=not node.name.startswith("_"),
            complexity_score=self._calculate_complexity(node),
        )

    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """Extract source code for a node."""
        lines = source.splitlines()
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        return "\n".join(lines[start:end])

    def _extract_dependencies(self, node: ast.AST) -> list[str]:
        """Extract function/method calls from a node."""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(child.func.attr)
        return list(set(dependencies))

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate cyclomatic complexity estimate."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
```

---

## 6. Dependency Graph Builder

```python
# twinscribe/orchestrator/dependency.py

from twinscribe.orchestrator.models import ComponentInfo


class DependencyGraphBuilder:
    """
    Builds dependency graph from discovered components.

    Uses call information to establish dependencies between components.
    """

    def build(self, components: dict[str, ComponentInfo]) -> dict[str, set[str]]:
        """
        Build dependency graph.

        Returns:
            Dict mapping component_id to set of dependency component_ids
        """
        graph = {cid: set() for cid in components}

        # Build reverse index of short names to full IDs
        name_to_ids = {}
        for cid in components:
            short_name = cid.split(".")[-1]
            if short_name not in name_to_ids:
                name_to_ids[short_name] = []
            name_to_ids[short_name].append(cid)

        # Resolve dependencies
        for cid, comp in components.items():
            for dep_name in comp.dependencies:
                # Try to resolve to a known component
                if dep_name in name_to_ids:
                    # If multiple matches, prefer same module
                    matches = name_to_ids[dep_name]
                    if len(matches) == 1:
                        graph[cid].add(matches[0])
                    else:
                        # Same module preference
                        module = ".".join(cid.split(".")[:-1])
                        for match in matches:
                            if match.startswith(module):
                                graph[cid].add(match)
                                break

        return graph
```

---

## 7. State Diagram

```
                          +-------+
                          | IDLE  |
                          +---+---+
                              |
                              | run()
                              v
                    +---------+---------+
                    | INITIALIZING      |
                    | - discover        |
                    | - build graph     |
                    | - init oracle     |
                    | - init streams    |
                    +---------+---------+
                              |
                              v
            +<----------------+
            |
            v         +-----------------+
    +-------+-------+ |                 |
    | PROCESSING    | |  WAITING_FOR_   |
    | - stream A    | |  BEADS          |
    | - stream B    | |  - create       |
    +-------+-------+ |  - wait         |
            |         |  - apply        |
            v         +--------+--------+
    +-------+-------+          ^
    | COMPARING     |          |
    | - compare     |          |
    | - check conv  +----------+
    +-------+-------+    needs_beads
            |
            | converged OR max_iter
            v
    +-------+-------+
    | FINALIZING    |
    | - merge       |
    | - call graph  |
    | - tickets     |
    | - report      |
    +-------+-------+
            |
            v
    +-------+-------+
    | COMPLETED     |
    +---------------+
```

---

## 8. Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Codebase ───► ComponentDiscovery ───► Components                     │
│                                             │                          │
│                                             ▼                          │
│                                    DependencyGraphBuilder              │
│                                             │                          │
│                                             ▼                          │
│                                    Topological Sort ───► ProcessingOrder│
│                                                                        │
│   Codebase ───► StaticAnalysisOracle ───► Ground Truth Call Graph      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         ITERATION LOOP                                  │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ProcessingOrder                                                      │
│        │                                                               │
│        ├─────────────────────────────────────┐                        │
│        │                                     │                         │
│        ▼                                     ▼                         │
│   ┌─────────────┐                     ┌─────────────┐                  │
│   │  Stream A   │                     │  Stream B   │                  │
│   │  (parallel) │                     │  (parallel) │                  │
│   └──────┬──────┘                     └──────┬──────┘                  │
│          │                                   │                         │
│          └──────────────┬────────────────────┘                         │
│                         │                                              │
│                         ▼                                              │
│                 ┌───────────────┐                                      │
│                 │ ComparatorAgent│◄─── Ground Truth                    │
│                 └───────┬───────┘                                      │
│                         │                                              │
│          ┌──────────────┼──────────────┐                              │
│          │              │              │                               │
│          ▼              ▼              ▼                               │
│    Corrections    Corrections    BeadsTickets                          │
│    (Stream A)     (Stream B)     (create/wait)                         │
│          │              │              │                               │
│          └──────────────┴──────────────┘                               │
│                         │                                              │
│                         ▼                                              │
│                  ConvergenceCheck                                      │
│                         │                                              │
│            ┌────────────┼────────────┐                                │
│            │            │            │                                 │
│            ▼            ▼            ▼                                 │
│        converged   max_iter     continue                              │
│            │            │            │                                 │
│            └────────────┤            └──► (loop back)                 │
│                         │                                              │
└─────────────────────────┼──────────────────────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         FINALIZATION                                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Stream A Output ─┬─► Merge Logic ───► Final Documentation            │
│   Stream B Output ─┘        │                                          │
│                             │                                          │
│                             ├───► Call Graph Extraction                │
│                             │                                          │
│                             ├───► Rebuild Ticket Generation            │
│                             │                                          │
│                             └───► Convergence Report                   │
│                                        │                               │
│                                        ▼                               │
│                               DocumentationPackage                     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Error Handling Strategy

```python
# twinscribe/orchestrator/exceptions.py

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class InitializationError(OrchestratorError):
    """Raised when initialization fails."""
    pass


class DiscoveryError(OrchestratorError):
    """Raised when component discovery fails."""
    pass


class StreamProcessingError(OrchestratorError):
    """Raised when stream processing fails."""
    pass


class ComparisonError(OrchestratorError):
    """Raised when comparison fails."""
    pass


class ConvergenceError(OrchestratorError):
    """Raised when convergence cannot be achieved."""
    pass


class FinalizationError(OrchestratorError):
    """Raised when finalization fails."""
    pass


# Error recovery strategies
class ErrorRecoveryStrategy:
    """Strategies for recovering from errors."""

    @staticmethod
    async def retry_stream(
        stream: "DocumentationStream",
        components: list[dict],
        max_retries: int = 3
    ):
        """Retry stream processing with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await stream.process(components)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise StreamProcessingError(f"Stream failed after {max_retries} retries: {e}")
                await asyncio.sleep(2 ** attempt)

    @staticmethod
    async def partial_results(
        stream_a_results: Optional[dict],
        stream_b_results: Optional[dict],
    ) -> dict:
        """Return partial results when one stream fails."""
        if stream_a_results and stream_b_results:
            return stream_a_results, stream_b_results
        elif stream_a_results:
            return stream_a_results, {}
        elif stream_b_results:
            return {}, stream_b_results
        else:
            raise StreamProcessingError("Both streams failed")
```

---

## 10. Configuration Schema

```yaml
# config.yaml

orchestrator:
  codebase_path: /path/to/codebase
  language: python
  max_iterations: 5
  parallel_streams: true
  beads_timeout_hours: 48
  force_converge_on_timeout: true

  exclude_patterns:
    - "**/test_*"
    - "**/tests/**"
    - "**/__pycache__/**"
    - "**/.git/**"
    - "**/migrations/**"

  include_private: false
  min_complexity_score: 0.0

agents:
  stream_a:
    documenter:
      provider: anthropic
      model: claude-sonnet-4-5-20250929
      max_tokens: 4096
      temperature: 0.3
    validator:
      provider: anthropic
      model: claude-haiku-4-5-20251001
      max_tokens: 2048
      temperature: 0.2

  stream_b:
    documenter:
      provider: openai
      model: gpt-4o
      max_tokens: 4096
      temperature: 0.3
    validator:
      provider: openai
      model: gpt-4o-mini
      max_tokens: 2048
      temperature: 0.2

  comparator:
    provider: anthropic
    model: claude-opus-4-5-20251101
    max_tokens: 8192
    temperature: 0.2

convergence:
  call_graph_match_rate: 0.98
  documentation_similarity: 0.95
  max_open_discrepancies: 2

beads:
  server: https://your-org.atlassian.net
  project: LEGACY_DOC
  rebuild_project: REBUILD

static_analysis:
  python:
    primary: pycg
    fallback: pyan3

output:
  documentation_path: ./output/documentation.json
  call_graph_path: ./output/call_graph.json
  rebuild_tickets_path: ./output/rebuild_tickets.json
  convergence_report_path: ./output/convergence_report.json
```

---

## 11. Interface Contracts Summary

| Interface | Input | Output | Description |
|-----------|-------|--------|-------------|
| `DualStreamOrchestrator.run()` | None | `DocumentationPackage` | Full pipeline execution |
| `DualStreamOrchestrator.get_state()` | None | `OrchestratorState` | Current state |
| `DualStreamOrchestrator.get_progress()` | None | `dict` | Progress info |
| `ComponentDiscovery.discover()` | `exclude_patterns, include_private` | `dict[str, ComponentInfo]` | Find components |
| `DependencyGraphBuilder.build()` | `components` | `dict[str, set[str]]` | Build dep graph |

---

## 12. Testing Strategy

### Unit Tests
- Test component discovery with mock files
- Test topological sort with known graphs
- Test merge logic with various scenarios
- Test state transitions

### Integration Tests
- Test full pipeline with small codebase
- Test parallel stream coordination
- Test Beads integration (mock)
- Test recovery from failures

### End-to-End Tests
- Full run on sample codebase
- Verify output format
- Verify convergence behavior

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Systems Architect | Initial design |
