"""
Orchestrator State Management.

Provides state persistence, checkpointing, and recovery for the orchestrator.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    """A checkpoint of orchestrator state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        created_at: When checkpoint was created
        iteration: Iteration number at checkpoint
        phase: Phase at checkpoint
        component_states: Per-component state
        stream_a_state: Stream A state
        stream_b_state: Stream B state
        comparison_results: Comparison results so far
        beads_tickets: Beads ticket state
    """

    checkpoint_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    iteration: int = 0
    phase: str = "not_started"
    component_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stream_a_state: dict[str, Any] = Field(default_factory=dict)
    stream_b_state: dict[str, Any] = Field(default_factory=dict)
    comparison_results: list[dict[str, Any]] = Field(default_factory=list)
    beads_tickets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointManager:
    """Manages orchestrator checkpoints for persistence and recovery.

    Provides:
    - Automatic checkpoint creation at iteration boundaries
    - Manual checkpoint creation
    - Recovery from checkpoints
    - Checkpoint cleanup

    Usage:
        manager = CheckpointManager(checkpoint_dir="./checkpoints")

        # Save checkpoint
        checkpoint_id = await manager.create_checkpoint(orchestrator)

        # List checkpoints
        checkpoints = manager.list_checkpoints()

        # Recover from checkpoint
        state = manager.load_checkpoint(checkpoint_id)
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 10,
        auto_checkpoint: bool = True,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            max_checkpoints: Maximum checkpoints to retain
            auto_checkpoint: Whether to auto-checkpoint at iterations
        """
        self._checkpoint_dir = Path(checkpoint_dir)
        self._max_checkpoints = max_checkpoints
        self._auto_checkpoint = auto_checkpoint

        # Ensure directory exists
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory."""
        return self._checkpoint_dir

    def create_checkpoint(
        self,
        orchestrator: "DualStreamOrchestrator",
        checkpoint_id: str | None = None,
    ) -> str:
        """Create a checkpoint from current orchestrator state.

        Args:
            orchestrator: Orchestrator instance
            checkpoint_id: Optional custom checkpoint ID

        Returns:
            Checkpoint ID
        """
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        state = orchestrator.state

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            iteration=state.iteration,
            phase=state.phase.value,
            component_states=self._extract_component_states(orchestrator),
            stream_a_state=self._extract_stream_state(orchestrator._stream_a),
            stream_b_state=self._extract_stream_state(orchestrator._stream_b),
            comparison_results=[asdict(r) for r in orchestrator.iteration_history],
            beads_tickets=self._extract_beads_state(orchestrator),
            metadata={
                "total_components": state.total_components,
                "processed_components": state.processed_components,
                "errors": state.errors,
            },
        )

        # Save to file
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_id}.json"
        with open(checkpoint_path, "w") as f:
            f.write(checkpoint.model_dump_json(indent=2))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """Load a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint to load

        Returns:
            Checkpoint data

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        with open(checkpoint_path) as f:
            data = json.load(f)

        return Checkpoint(**data)

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints.

        Returns:
            List of checkpoint IDs, newest first
        """
        checkpoints = []

        for path in self._checkpoint_dir.glob("checkpoint_*.json"):
            checkpoints.append(path.stem)

        # Sort by timestamp (newest first)
        checkpoints.sort(reverse=True)

        return checkpoints

    def get_latest_checkpoint(self) -> str | None:
        """Get the most recent checkpoint ID.

        Returns:
            Latest checkpoint ID or None
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_id}.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            return True
        return False

    def _extract_component_states(
        self,
        orchestrator: "DualStreamOrchestrator",
    ) -> dict[str, dict[str, Any]]:
        """Extract per-component states.

        Args:
            orchestrator: Orchestrator instance

        Returns:
            Component states dictionary
        """
        states = {}

        for comp in orchestrator._components:
            states[comp.id] = {
                "processed": comp.id in orchestrator._component_results,
                "converged": False,  # Would check convergence
            }

        return states

    def _extract_stream_state(
        self,
        stream: "DocumentationStream",
    ) -> dict[str, Any]:
        """Extract stream state.

        Args:
            stream: Documentation stream

        Returns:
            Stream state dictionary
        """
        try:
            outputs = stream.get_outputs()
            return {
                "component_count": len(outputs),
                "component_ids": list(outputs.keys()),
            }
        except Exception:
            return {}

    def _extract_beads_state(
        self,
        orchestrator: "DualStreamOrchestrator",
    ) -> dict[str, dict[str, Any]]:
        """Extract Beads ticket state.

        Args:
            orchestrator: Orchestrator instance

        Returns:
            Beads state dictionary
        """
        if not orchestrator._beads_manager:
            return {}

        tracker = orchestrator._beads_manager.tracker
        return tracker.to_dict()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max_checkpoints."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self._max_checkpoints:
            for old_checkpoint in checkpoints[self._max_checkpoints :]:
                self.delete_checkpoint(old_checkpoint)


class StateRecovery:
    """Recovers orchestrator state from checkpoints.

    Handles:
    - Partial iteration recovery
    - Stream state restoration
    - Beads ticket synchronization
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        """Initialize state recovery.

        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self._manager = checkpoint_manager

    async def recover(
        self,
        orchestrator: "DualStreamOrchestrator",
        checkpoint_id: str | None = None,
    ) -> bool:
        """Recover orchestrator state from checkpoint.

        Args:
            orchestrator: Orchestrator to restore
            checkpoint_id: Checkpoint to restore from (latest if None)

        Returns:
            True if recovery successful, False otherwise
        """
        # Get checkpoint
        if checkpoint_id is None:
            checkpoint_id = self._manager.get_latest_checkpoint()

        if not checkpoint_id:
            return False

        try:
            checkpoint = self._manager.load_checkpoint(checkpoint_id)
        except FileNotFoundError:
            return False

        # Restore state
        orchestrator._state.iteration = checkpoint.iteration
        orchestrator._state.phase = checkpoint.phase

        # Restore stream states
        await self._restore_stream_state(
            orchestrator._stream_a,
            checkpoint.stream_a_state,
        )
        await self._restore_stream_state(
            orchestrator._stream_b,
            checkpoint.stream_b_state,
        )

        # Restore Beads state
        if orchestrator._beads_manager and checkpoint.beads_tickets:
            await self._restore_beads_state(
                orchestrator._beads_manager,
                checkpoint.beads_tickets,
            )

        return True

    async def _restore_stream_state(
        self,
        stream: "DocumentationStream",
        state: dict[str, Any],
    ) -> None:
        """Restore stream state.

        Args:
            stream: Stream to restore
            state: State to restore from
        """
        # Implementation depends on stream interface
        # Would restore cached outputs, etc.
        pass

    async def _restore_beads_state(
        self,
        beads_manager: "BeadsLifecycleManager",
        state: dict[str, Any],
    ) -> None:
        """Restore Beads state.

        Args:
            beads_manager: Beads manager to restore
            state: State to restore from
        """
        from twinscribe.beads.tracker import TicketTracker

        # Restore tracker from serialized state
        restored_tracker = TicketTracker.from_dict(state)

        # Copy state to manager's tracker
        beads_manager._tracker = restored_tracker

        # Sync with actual Beads system to update ticket statuses
        # This handles tickets that were resolved while offline
        await beads_manager.sync_from_beads(
            f'project = "{beads_manager.config.project}" AND labels = "ai-documentation"'
        )


class ProgressTracker:
    """Tracks detailed progress for the orchestrator.

    Provides:
    - Component-level progress tracking
    - Time estimates
    - Throughput metrics
    """

    def __init__(self) -> None:
        """Initialize progress tracker."""
        self._start_time: datetime | None = None
        self._component_times: dict[str, float] = {}
        self._iteration_times: list[float] = []

    def start(self) -> None:
        """Start progress tracking."""
        self._start_time = datetime.utcnow()

    def record_component(self, component_id: str, duration_seconds: float) -> None:
        """Record component processing time.

        Args:
            component_id: Component processed
            duration_seconds: Processing duration
        """
        self._component_times[component_id] = duration_seconds

    def record_iteration(self, duration_seconds: float) -> None:
        """Record iteration time.

        Args:
            duration_seconds: Iteration duration
        """
        self._iteration_times.append(duration_seconds)

    def estimate_remaining(
        self,
        remaining_components: int,
        remaining_iterations: int,
    ) -> float | None:
        """Estimate remaining time.

        Args:
            remaining_components: Components left to process
            remaining_iterations: Iterations left

        Returns:
            Estimated seconds remaining or None
        """
        if not self._component_times:
            return None

        avg_component_time = sum(self._component_times.values()) / len(self._component_times)

        if self._iteration_times:
            avg_iteration_overhead = sum(self._iteration_times) / len(self._iteration_times)
        else:
            avg_iteration_overhead = 0

        return (
            remaining_components * avg_component_time
            + remaining_iterations * avg_iteration_overhead
        )

    def get_throughput(self) -> float | None:
        """Get components per second throughput.

        Returns:
            Components per second or None
        """
        if not self._start_time or not self._component_times:
            return None

        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        if elapsed == 0:
            return None

        return len(self._component_times) / elapsed

    def get_summary(self) -> dict[str, Any]:
        """Get progress summary.

        Returns:
            Summary dictionary
        """
        elapsed = None
        if self._start_time:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "elapsed_seconds": elapsed,
            "components_processed": len(self._component_times),
            "iterations_completed": len(self._iteration_times),
            "throughput_per_second": self.get_throughput(),
            "avg_component_time": (
                sum(self._component_times.values()) / len(self._component_times)
                if self._component_times
                else None
            ),
            "avg_iteration_time": (
                sum(self._iteration_times) / len(self._iteration_times)
                if self._iteration_times
                else None
            ),
        }
