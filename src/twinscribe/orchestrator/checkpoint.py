"""
Checkpoint Manager for State Persistence.

Provides JSONL append-only checkpoint logging for tracking documentation
pipeline progress. Each event is appended as a single line, enabling:
- Crash recovery by replaying the log
- Progress monitoring
- Audit trail of processing

Checkpoint format uses JSONL with event types:
- run_start: Pipeline initialization
- discovery_complete: Component discovery finished
- component_documented: Single component completed
- error: Error occurred during processing
- iteration_complete: Full iteration finished
- run_complete: Pipeline finished
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class CheckpointEvent:
    """Base class for checkpoint events.

    All events share common fields for identification and timing.

    Attributes:
        type: Event type identifier
        timestamp: ISO format timestamp
        run_id: Unique run identifier
    """

    type: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class RunStartEvent(CheckpointEvent):
    """Event logged when a documentation run starts.

    Attributes:
        config: Configuration used for the run
    """

    type: str = "run_start"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryCompleteEvent(CheckpointEvent):
    """Event logged when component discovery finishes.

    Attributes:
        component_count: Number of components discovered
        component_ids: List of component IDs
        processing_order: Topological order for processing
    """

    type: str = "discovery_complete"
    component_count: int = 0
    component_ids: list[str] = field(default_factory=list)
    processing_order: list[str] = field(default_factory=list)


@dataclass
class ComponentDocumentedEvent(CheckpointEvent):
    """Event logged when a component is successfully documented.

    Attributes:
        component_id: ID of the documented component
        stream: Stream identifier (A or B)
        iteration: Current iteration number
        status: Processing status (success, failed, skipped)
        output_path: Path to the saved output file
        duration_ms: Processing duration in milliseconds
        token_count: Tokens used (optional)
    """

    type: str = "component_documented"
    component_id: str = ""
    stream: str = ""
    iteration: int = 1
    status: str = "success"
    output_path: str | None = None
    duration_ms: float = 0.0
    token_count: int | None = None


@dataclass
class ErrorEvent(CheckpointEvent):
    """Event logged when an error occurs.

    Attributes:
        phase: Processing phase where error occurred
        component_id: Component being processed (if applicable)
        stream: Stream identifier (if applicable)
        error: Error message
        error_type: Exception type name
        traceback: Optional truncated traceback
    """

    type: str = "error"
    phase: str = ""
    component_id: str | None = None
    stream: str | None = None
    error: str = ""
    error_type: str = ""
    traceback: str | None = None


@dataclass
class IterationCompleteEvent(CheckpointEvent):
    """Event logged when an iteration completes.

    Attributes:
        iteration: Iteration number
        components_processed: Number of components processed
        discrepancies_found: Number of discrepancies detected
        discrepancies_resolved: Number of discrepancies resolved
        converged: Whether convergence was achieved
        duration_seconds: Total iteration duration
    """

    type: str = "iteration_complete"
    iteration: int = 0
    components_processed: int = 0
    discrepancies_found: int = 0
    discrepancies_resolved: int = 0
    converged: bool = False
    duration_seconds: float = 0.0


@dataclass
class RunCompleteEvent(CheckpointEvent):
    """Event logged when a run completes.

    Attributes:
        status: Final status (completed, failed, cancelled)
        total_iterations: Number of iterations performed
        total_components: Total components documented
        total_discrepancies: Total discrepancies found
        total_duration_seconds: Total run duration
        metrics: Additional metrics dictionary
    """

    type: str = "run_complete"
    status: str = "completed"
    total_iterations: int = 0
    total_components: int = 0
    total_discrepancies: int = 0
    total_duration_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def _sanitize_component_id(component_id: str) -> str:
    """Sanitize component ID for use as a filename.

    Replaces unsafe characters with underscores and limits length.

    Args:
        component_id: Original component identifier

    Returns:
        Safe filename string
    """
    # Replace path separators, dots, and other unsafe chars
    safe = re.sub(r"[<>:\"/\\|?*.]", "_", component_id)
    # Collapse multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Limit length (preserve end which is usually the method name)
    if len(safe) > 200:
        safe = safe[-200:]
    return safe.strip("_")


class CheckpointManager:
    """Manages JSONL append-only checkpoints for state persistence.

    This manager provides crash-safe state persistence by:
    1. Writing events to an append-only JSONL file
    2. Using atomic writes (tmp + rename) for reliability
    3. Saving component outputs to individual files
    4. Supporting recovery by replaying the checkpoint log

    The checkpoint file format is one JSON object per line:
    ```
    {"type":"run_start","run_id":"abc123","timestamp":"...","config":{...}}
    {"type":"component_documented","component_id":"foo.bar","stream":"A",...}
    ```

    Component outputs are saved separately to avoid huge checkpoint files:
    ```
    output/components/{component_id_safe}.a.json
    output/components/{component_id_safe}.b.json
    ```

    Usage:
        manager = CheckpointManager(output_dir="./output")
        manager.record_run_start(config)
        manager.record_component_documented(
            component_id="module.func",
            stream_id="A",
            iteration=1,
            output=documentation_output,
        )

    Attributes:
        output_dir: Base directory for all output files
        run_id: Unique identifier for this run
    """

    def __init__(
        self,
        output_dir: str | Path = "./output",
        run_id: str | None = None,
    ) -> None:
        """Initialize the checkpoint manager.

        Args:
            output_dir: Base directory for checkpoints and component outputs
            run_id: Optional run identifier (auto-generated if not provided)
        """
        self._output_dir = Path(output_dir)
        self._run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

        # Create directory structure
        self._checkpoint_dir = self._output_dir / "checkpoints"
        self._component_dir = self._output_dir / "components"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._component_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file path
        self._checkpoint_path = self._checkpoint_dir / f"checkpoint_{self._run_id}.jsonl"

        logger.info(f"CheckpointManager initialized: run_id={self._run_id}")

    @property
    def run_id(self) -> str:
        """Get the current run identifier."""
        return self._run_id

    @property
    def checkpoint_path(self) -> Path:
        """Get the path to the checkpoint file."""
        return self._checkpoint_path

    @property
    def component_dir(self) -> Path:
        """Get the component output directory."""
        return self._component_dir

    def _write_event(self, event: CheckpointEvent) -> None:
        """Write an event to the checkpoint file atomically.

        Uses tmp file + rename pattern for crash safety.

        Args:
            event: Event to write
        """
        event.run_id = self._run_id
        event.timestamp = datetime.utcnow().isoformat()

        line = event.to_json() + "\n"

        # For JSONL append, we use atomic append via a helper
        self._atomic_append(self._checkpoint_path, line)

        logger.debug(f"Recorded checkpoint event: {event.type}")

    def _atomic_append(self, path: Path, content: str) -> None:
        """Atomically append content to a file.

        For truly atomic appends, we:
        1. Read existing content (if any)
        2. Write to a temp file with all content
        3. Rename temp to final

        This is more conservative but ensures no partial writes.

        Args:
            path: Target file path
            content: Content to append
        """
        tmp_path = path.with_suffix(".tmp")

        try:
            # Read existing content
            existing = ""
            if path.exists():
                existing = path.read_text(encoding="utf-8")

            # Write combined content to temp file
            tmp_path.write_text(existing + content, encoding="utf-8")

            # Atomic rename
            os.replace(tmp_path, path)

        except Exception as e:
            # Clean up temp file on error
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to write checkpoint: {e}")
            raise

    def _atomic_write(self, path: Path, content: str) -> None:
        """Atomically write content to a file.

        Uses tmp file + rename pattern for crash safety.

        Args:
            path: Target file path
            content: Content to write
        """
        tmp_path = path.with_suffix(path.suffix + ".tmp")

        try:
            # Write to temp file
            tmp_path.write_text(content, encoding="utf-8")

            # Atomic rename
            os.replace(tmp_path, path)

        except Exception as e:
            # Clean up temp file on error
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to write file: {e}")
            raise

    def record_run_start(self, config: dict[str, Any] | None = None) -> None:
        """Record the start of a documentation run.

        Args:
            config: Configuration dictionary for the run
        """
        event = RunStartEvent(
            run_id=self._run_id,
            config=config or {},
        )
        self._write_event(event)
        logger.info(f"Recorded run start: {self._run_id}")

    def record_discovery_complete(
        self,
        components: list[Any],
        processing_order: list[str],
    ) -> None:
        """Record completion of component discovery.

        Args:
            components: List of discovered components (with component_id attribute)
            processing_order: Topological processing order
        """
        component_ids = [getattr(c, "component_id", str(c)) for c in components]

        event = DiscoveryCompleteEvent(
            run_id=self._run_id,
            component_count=len(components),
            component_ids=component_ids,
            processing_order=processing_order,
        )
        self._write_event(event)
        logger.info(f"Recorded discovery complete: {len(components)} components")

    def record_component_documented(
        self,
        component_id: str,
        stream_id: str,
        iteration: int,
        output: Any,
        output_path: str | None = None,
        duration_ms: float = 0.0,
        token_count: int | None = None,
    ) -> str:
        """Record successful documentation of a component.

        Also saves the component output to a separate file.

        Args:
            component_id: ID of the documented component
            stream_id: Stream identifier (A or B)
            iteration: Current iteration number
            output: Documentation output object (Pydantic model or dict)
            output_path: Optional custom output path
            duration_ms: Processing duration in milliseconds
            token_count: Tokens used (optional)

        Returns:
            Path to the saved output file
        """
        # Determine output path
        if output_path is None:
            safe_id = _sanitize_component_id(component_id)
            stream_suffix = stream_id.lower()
            output_path = str(self._component_dir / f"{safe_id}.{stream_suffix}.json")

        # Save component output
        self._save_component_output(output, output_path)

        # Record event
        event = ComponentDocumentedEvent(
            run_id=self._run_id,
            component_id=component_id,
            stream=stream_id,
            iteration=iteration,
            status="success",
            output_path=output_path,
            duration_ms=duration_ms,
            token_count=token_count,
        )
        self._write_event(event)
        logger.debug(f"Recorded component documented: {component_id} ({stream_id})")

        return output_path

    def _save_component_output(self, output: Any, path: str) -> None:
        """Save component output to a file atomically.

        Args:
            output: Output object (Pydantic model, dataclass, or dict)
            path: File path to save to
        """
        # Convert output to JSON-serializable dict
        if hasattr(output, "model_dump"):
            # Pydantic v2
            data = output.model_dump(mode="json")
        elif hasattr(output, "dict"):
            # Pydantic v1
            data = output.dict()
        elif hasattr(output, "__dataclass_fields__"):
            # dataclass
            data = asdict(output)
        elif isinstance(output, dict):
            data = output
        else:
            # Fallback: try to convert to dict
            data = {"value": str(output)}

        content = json.dumps(data, indent=2, default=str)
        self._atomic_write(Path(path), content)

    def record_error(
        self,
        phase: str,
        component_id: str | None = None,
        stream_id: str | None = None,
        error: str | Exception = "",
        traceback: str | None = None,
    ) -> None:
        """Record an error that occurred during processing.

        Args:
            phase: Processing phase (discovering, documenting, comparing, etc.)
            component_id: Component being processed (if applicable)
            stream_id: Stream identifier (if applicable)
            error: Error message or exception
            traceback: Optional traceback string
        """
        error_msg = str(error)
        error_type = type(error).__name__ if isinstance(error, Exception) else "Error"

        event = ErrorEvent(
            run_id=self._run_id,
            phase=phase,
            component_id=component_id,
            stream=stream_id,
            error=error_msg,
            error_type=error_type,
            traceback=traceback,
        )
        self._write_event(event)
        logger.warning(
            f"Recorded error: phase={phase}, component={component_id}, error={error_msg}"
        )

    def record_iteration_complete(
        self,
        iteration: int,
        result: Any,
    ) -> None:
        """Record completion of an iteration.

        Args:
            iteration: Iteration number
            result: IterationResult object or dict with iteration metrics
        """
        # Extract fields from result object or dict
        if hasattr(result, "components_processed"):
            components_processed = result.components_processed
            discrepancies_found = getattr(result, "discrepancies_found", 0)
            discrepancies_resolved = getattr(result, "discrepancies_resolved", 0)
            converged = getattr(result, "converged", False)
            duration = getattr(result, "duration_seconds", 0.0)
        elif isinstance(result, dict):
            components_processed = result.get("components_processed", 0)
            discrepancies_found = result.get("discrepancies_found", 0)
            discrepancies_resolved = result.get("discrepancies_resolved", 0)
            converged = result.get("converged", False)
            duration = result.get("duration_seconds", 0.0)
        else:
            components_processed = 0
            discrepancies_found = 0
            discrepancies_resolved = 0
            converged = False
            duration = 0.0

        event = IterationCompleteEvent(
            run_id=self._run_id,
            iteration=iteration,
            components_processed=components_processed,
            discrepancies_found=discrepancies_found,
            discrepancies_resolved=discrepancies_resolved,
            converged=converged,
            duration_seconds=duration,
        )
        self._write_event(event)
        logger.info(f"Recorded iteration complete: {iteration}")

    def record_run_complete(
        self,
        status: str,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record completion of a documentation run.

        Args:
            status: Final status (completed, failed, cancelled)
            metrics: Final metrics dictionary
        """
        metrics = metrics or {}

        event = RunCompleteEvent(
            run_id=self._run_id,
            status=status,
            total_iterations=metrics.get("total_iterations", 0),
            total_components=metrics.get("total_components", 0),
            total_discrepancies=metrics.get("total_discrepancies", 0),
            total_duration_seconds=metrics.get("total_duration_seconds", 0.0),
            metrics=metrics,
        )
        self._write_event(event)
        logger.info(f"Recorded run complete: status={status}")

    def load_checkpoint(self) -> list[dict[str, Any]]:
        """Load all events from the checkpoint file.

        Returns:
            List of event dictionaries in order
        """
        events = []

        if not self._checkpoint_path.exists():
            return events

        with open(self._checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse checkpoint line: {e}")

        return events

    def get_processed_components(self) -> dict[str, set[str]]:
        """Get components that have been successfully processed.

        Returns:
            Dictionary mapping stream ID to set of component IDs
        """
        events = self.load_checkpoint()
        processed: dict[str, set[str]] = {"A": set(), "B": set()}

        for event in events:
            if event.get("type") == "component_documented":
                stream = event.get("stream", "")
                component_id = event.get("component_id", "")
                if stream in processed and component_id:
                    processed[stream].add(component_id)

        return processed

    def get_last_iteration(self) -> int:
        """Get the last completed iteration number.

        Returns:
            Last iteration number or 0 if none completed
        """
        events = self.load_checkpoint()
        last_iteration = 0

        for event in events:
            if event.get("type") == "iteration_complete":
                iteration = event.get("iteration", 0)
                if iteration > last_iteration:
                    last_iteration = iteration

        return last_iteration

    def get_run_status(self) -> str | None:
        """Get the final run status if completed.

        Returns:
            Status string or None if not completed
        """
        events = self.load_checkpoint()

        for event in reversed(events):
            if event.get("type") == "run_complete":
                return event.get("status")

        return None

    def load_component_output(
        self,
        component_id: str,
        stream_id: str,
    ) -> dict[str, Any] | None:
        """Load a saved component output.

        Args:
            component_id: Component identifier
            stream_id: Stream identifier (A or B)

        Returns:
            Output dictionary or None if not found
        """
        safe_id = _sanitize_component_id(component_id)
        stream_suffix = stream_id.lower()
        path = self._component_dir / f"{safe_id}.{stream_suffix}.json"

        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load component output: {e}")
            return None

    @classmethod
    def find_resumable_runs(
        cls,
        checkpoint_dir: str | Path,
        codebase_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Find incomplete runs that can be resumed.

        Scans the checkpoint directory for runs that started but did not
        complete (no run_complete event).

        Args:
            checkpoint_dir: Directory containing checkpoint files
            codebase_path: If provided, only return runs for this codebase

        Returns:
            List of resumable run info dicts with keys:
                - run_id: The run identifier
                - checkpoint_path: Path to the checkpoint file
                - started_at: Run start timestamp
                - last_event_at: Last event timestamp
                - components_processed: Number of components documented
                - last_iteration: Last completed iteration
                - config: Original run configuration
        """
        checkpoint_path = Path(checkpoint_dir)
        resumable = []

        if not checkpoint_path.exists():
            return resumable

        # Find all checkpoint files
        for checkpoint_file in checkpoint_path.glob("checkpoint_*.jsonl"):
            try:
                events = []
                with open(checkpoint_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                if not events:
                    continue

                # Check if run completed successfully (failed runs are resumable)
                run_complete_event = next(
                    (e for e in events if e.get("type") == "run_complete"),
                    None,
                )

                if run_complete_event:
                    run_status = run_complete_event.get("status", "")
                    if run_status != "failed":
                        # Run completed successfully, not resumable
                        continue

                # Extract run info
                run_start = next(
                    (e for e in events if e.get("type") == "run_start"),
                    None,
                )
                if not run_start:
                    continue

                run_id = run_start.get("run_id", "")

                # Count processed components
                processed_components: dict[str, set[str]] = {"A": set(), "B": set()}
                for event in events:
                    if event.get("type") == "component_documented":
                        stream = event.get("stream", "")
                        comp_id = event.get("component_id", "")
                        if stream in processed_components and comp_id:
                            processed_components[stream].add(comp_id)

                # Get last iteration
                last_iteration = 0
                for event in events:
                    if event.get("type") == "iteration_complete":
                        iteration = event.get("iteration", 0)
                        if iteration > last_iteration:
                            last_iteration = iteration

                # Get timestamps
                started_at = run_start.get("timestamp", "")
                last_event_at = events[-1].get("timestamp", "") if events else ""

                # Total unique components across both streams
                all_components = processed_components["A"] | processed_components["B"]

                resumable.append(
                    {
                        "run_id": run_id,
                        "checkpoint_path": str(checkpoint_file),
                        "started_at": started_at,
                        "last_event_at": last_event_at,
                        "components_processed": len(all_components),
                        "stream_a_processed": len(processed_components["A"]),
                        "stream_b_processed": len(processed_components["B"]),
                        "last_iteration": last_iteration,
                        "config": run_start.get("config", {}),
                    }
                )

            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to read checkpoint file {checkpoint_file}: {e}")
                continue

        # Filter by codebase_path if provided
        if codebase_path is not None:
            target_path = Path(codebase_path).resolve()
            filtered = []
            for run in resumable:
                run_codebase = run.get("config", {}).get("codebase_path", "")
                if run_codebase:
                    run_path = Path(run_codebase).resolve()
                    if run_path == target_path:
                        filtered.append(run)
            resumable = filtered

        # Sort by components processed (most progress first), then by time
        # This ensures runs with actual work are preferred over empty/new runs
        resumable.sort(
            key=lambda r: (r.get("components_processed", 0), r.get("last_event_at", "")),
            reverse=True,
        )

        return resumable


class CheckpointState(BaseModel):
    """Reconstructed state from checkpoint events for resuming a run.

    This class holds the state needed to resume a documentation run
    from a checkpoint. It extracts and organizes information from
    the checkpoint event log.

    Attributes:
        run_id: The original run identifier
        config: Original run configuration
        last_iteration: Last completed iteration number
        current_iteration: Current iteration (may be incomplete)
        processed_components: Dict mapping stream ID to set of processed component IDs
        component_outputs: Dict mapping (component_id, stream) to output path
        discovery_info: Component discovery information if available
        errors: List of recorded errors
        is_complete: Whether the run completed
    """

    run_id: str = Field(..., description="Original run identifier")
    config: dict[str, Any] = Field(default_factory=dict, description="Original configuration")
    last_iteration: int = Field(default=0, description="Last completed iteration")
    current_iteration: int = Field(default=1, description="Current/next iteration to run")
    processed_components: dict[str, list[str]] = Field(
        default_factory=lambda: {"A": [], "B": []},
        description="Stream ID to list of processed component IDs",
    )
    component_outputs: dict[str, str] = Field(
        default_factory=dict,
        description="(component_id:stream) to output path mapping",
    )
    discovery_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Component discovery information",
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recorded errors",
    )
    is_complete: bool = Field(default=False, description="Whether run completed")

    def get_processed_set(self, stream_id: str) -> set[str]:
        """Get set of processed component IDs for a stream.

        Args:
            stream_id: Stream identifier (A or B)

        Returns:
            Set of component IDs that have been processed
        """
        return set(self.processed_components.get(stream_id, []))

    def is_component_processed(self, component_id: str, stream_id: str) -> bool:
        """Check if a component has been processed in a stream.

        Args:
            component_id: Component identifier
            stream_id: Stream identifier (A or B)

        Returns:
            True if the component was already processed
        """
        return component_id in self.get_processed_set(stream_id)

    def get_output_path(self, component_id: str, stream_id: str) -> str | None:
        """Get the output path for a processed component.

        Args:
            component_id: Component identifier
            stream_id: Stream identifier (A or B)

        Returns:
            Path to output file or None if not found
        """
        key = f"{component_id}:{stream_id}"
        return self.component_outputs.get(key)

    @classmethod
    def build_state(cls, checkpoint_manager: CheckpointManager) -> CheckpointState:
        """Build checkpoint state from a checkpoint manager.

        Reconstructs the state by replaying all events from the checkpoint
        file. This allows resuming from where the run left off.

        Args:
            checkpoint_manager: Checkpoint manager with loaded checkpoint

        Returns:
            CheckpointState instance representing the current state
        """
        events = checkpoint_manager.load_checkpoint()

        # Initialize state
        run_id = checkpoint_manager.run_id
        config: dict[str, Any] = {}
        last_iteration = 0
        current_iteration = 1
        processed_components: dict[str, set[str]] = {"A": set(), "B": set()}
        component_outputs: dict[str, str] = {}
        discovery_info: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        is_complete = False

        # Replay events
        for event in events:
            event_type = event.get("type", "")

            if event_type == "run_start":
                run_id = event.get("run_id", run_id)
                config = event.get("config", {})

            elif event_type == "discovery_complete":
                discovery_info = {
                    "component_count": event.get("component_count", 0),
                    "component_ids": event.get("component_ids", []),
                    "processing_order": event.get("processing_order", []),
                }

            elif event_type == "component_documented":
                stream = event.get("stream", "")
                comp_id = event.get("component_id", "")
                output_path = event.get("output_path", "")
                iteration = event.get("iteration", 1)

                if stream in processed_components and comp_id:
                    processed_components[stream].add(comp_id)

                if comp_id and output_path:
                    key = f"{comp_id}:{stream}"
                    component_outputs[key] = output_path

                # Track current iteration based on component events
                if iteration > current_iteration:
                    current_iteration = iteration

            elif event_type == "iteration_complete":
                iteration = event.get("iteration", 0)
                if iteration > last_iteration:
                    last_iteration = iteration
                # Next iteration starts after this
                current_iteration = last_iteration + 1

            elif event_type == "error":
                errors.append(
                    {
                        "phase": event.get("phase", ""),
                        "component_id": event.get("component_id"),
                        "stream": event.get("stream"),
                        "error": event.get("error", ""),
                        "error_type": event.get("error_type", ""),
                    }
                )

            elif event_type == "run_complete":
                is_complete = True

        # Convert sets to lists for Pydantic serialization
        processed_lists = {
            stream: list(comp_ids) for stream, comp_ids in processed_components.items()
        }

        return cls(
            run_id=run_id,
            config=config,
            last_iteration=last_iteration,
            current_iteration=current_iteration,
            processed_components=processed_lists,
            component_outputs=component_outputs,
            discovery_info=discovery_info,
            errors=errors,
            is_complete=is_complete,
        )
