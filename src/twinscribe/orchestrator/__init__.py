"""
Dual-Stream Documentation System - Orchestrator Module

This module coordinates the entire documentation pipeline:

- DualStreamOrchestrator: Main orchestrator coordinating all phases
- Convergence management: Criteria checking and progress tracking
- State management: Checkpointing and recovery
- Progress tracking: Metrics and time estimation

Key Workflow:
1. Initialization: Parse codebase, run static analysis, build dependency graph
2. Iteration Loop: Document -> Validate -> Compare -> Resolve -> Check Convergence
3. Finalization: Merge outputs, generate rebuild tickets, produce report

Usage:
    from twinscribe.orchestrator import (
        DualStreamOrchestrator,
        OrchestratorConfig,
        ConvergenceCriteria,
    )

    config = OrchestratorConfig(max_iterations=5)
    orchestrator = DualStreamOrchestrator(
        config=config,
        static_oracle=oracle,
        stream_a=stream_a,
        stream_b=stream_b,
        comparator=comparator,
    )

    result = await orchestrator.run()
"""

from twinscribe.orchestrator.orchestrator import (
    DualStreamOrchestrator,
    OrchestratorConfig,
    OrchestratorState,
    OrchestratorPhase,
    OrchestratorError,
    IterationResult,
    ProgressCallback,
)
from twinscribe.orchestrator.convergence import (
    ConvergenceCriteria,
    ConvergenceStatus,
    ConvergenceCheck,
    ConvergenceChecker,
    ConvergenceTracker,
    ConvergenceHistoryEntry,
    BlockingDiscrepancyType,
    calculate_similarity,
)
from twinscribe.orchestrator.state import (
    Checkpoint,
    CheckpointManager,
    StateRecovery,
    ProgressTracker,
)

__all__ = [
    # Orchestrator
    "DualStreamOrchestrator",
    "OrchestratorConfig",
    "OrchestratorState",
    "OrchestratorPhase",
    "OrchestratorError",
    "IterationResult",
    "ProgressCallback",
    # Convergence
    "ConvergenceCriteria",
    "ConvergenceStatus",
    "ConvergenceCheck",
    "ConvergenceChecker",
    "ConvergenceTracker",
    "ConvergenceHistoryEntry",
    "BlockingDiscrepancyType",
    "calculate_similarity",
    # State
    "Checkpoint",
    "CheckpointManager",
    "StateRecovery",
    "ProgressTracker",
]
