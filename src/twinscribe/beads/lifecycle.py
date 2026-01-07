"""
Beads Lifecycle Manager for Documentation Tasks.

Extends the core BeadsLifecycleManager with specific methods for managing
documentation workflow lifecycle:
- Creating documentation tickets when tasks start
- Updating ticket status through workflow phases
- Recording validation results
- Managing ticket dependencies
- Closing tickets with completion reasons

This module provides the integration layer between the DualStreamOrchestrator
and the Beads issue tracking system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from twinscribe.beads.client import (
    BeadsClient,
    BeadsClientConfig,
    BeadsError,
    CreateIssueRequest,
)
from twinscribe.beads.tracker import (
    TicketStatus,
    TicketTracker,
    TicketType,
    TrackedTicket,
)


class DocumentationTicketStatus(str, Enum):
    """Status values for documentation tickets."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    DOCUMENTING = "documenting"
    VALIDATING = "validating"
    COMPARING = "comparing"
    AWAITING_REVIEW = "awaiting_review"
    RESOLVED = "resolved"
    CLOSED = "closed"


class CloseReason(str, Enum):
    """Reasons for closing a documentation ticket."""

    COMPLETED = "completed"
    CONVERGED = "converged"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"
    FAILED = "failed"


@dataclass
class ConvergenceMetrics:
    """Convergence metrics for documentation tasks.

    Attributes:
        iteration: Current iteration number
        call_graph_match_rate: Match rate between streams for call graph
        documentation_similarity: Similarity score for documentation
        discrepancies_remaining: Count of unresolved discrepancies
        timestamp: When metrics were recorded
    """

    iteration: int = 0
    call_graph_match_rate: float = 0.0
    documentation_similarity: float = 0.0
    discrepancies_remaining: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration": self.iteration,
            "call_graph_match_rate": self.call_graph_match_rate,
            "documentation_similarity": self.documentation_similarity,
            "discrepancies_remaining": self.discrepancies_remaining,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationSummary:
    """Summary of validation results for ticket updates.

    Attributes:
        component_id: Component that was validated
        status: Validation pass/fail/warning
        completeness_score: Completeness check score
        call_graph_accuracy: Call graph accuracy score
        corrections_count: Number of corrections applied
        errors: List of validation errors
    """

    component_id: str
    status: str
    completeness_score: float = 1.0
    call_graph_accuracy: float = 1.0
    corrections_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "component_id": self.component_id,
            "status": self.status,
            "completeness_score": self.completeness_score,
            "call_graph_accuracy": self.call_graph_accuracy,
            "corrections_count": self.corrections_count,
            "errors": self.errors,
        }


class LifecycleManagerConfig(BaseModel):
    """Configuration for BeadsLifecycleManager.

    Attributes:
        beads_directory: Path to .beads directory
        default_labels: Labels to add to all tickets
        auto_sync: Whether to sync after each operation
        update_on_progress: Whether to update ticket on progress events
        max_retries: Maximum retry attempts for operations
    """

    beads_directory: str = Field(default=".beads")
    default_labels: list[str] = Field(default_factory=lambda: ["ai-documentation", "twinscribe"])
    auto_sync: bool = Field(default=True)
    update_on_progress: bool = Field(default=True)
    max_retries: int = Field(default=3, ge=0)


class BeadsLifecycleManager:
    """Manages the lifecycle of Beads tickets for documentation tasks.

    This class provides the integration between the DualStreamOrchestrator
    and the Beads issue tracking system. It handles:

    1. **Ticket Lifecycle Management**:
       - Creates tickets when documentation tasks start
       - Updates ticket status through workflow phases
       - Closes tickets with appropriate reasons

    2. **Progress Tracking**:
       - Records validation results as ticket updates
       - Tracks convergence metrics
       - Maintains history of documentation iterations

    3. **Dependency Management**:
       - Links related tickets (component dependencies)
       - Tracks blocking relationships
       - Manages dependency ordering

    Usage:
        manager = BeadsLifecycleManager(config)
        await manager.initialize()

        # Create documentation ticket
        ticket_id = await manager.create_documentation_ticket(component)

        # Update status during workflow
        await manager.update_ticket_status(ticket_id, "in_progress")

        # Record validation results
        await manager.record_validation_result(ticket_id, validation_result)

        # Link dependencies
        await manager.link_dependencies(ticket_id, ["dep-1", "dep-2"])

        # Close when done
        await manager.close_ticket(ticket_id, CloseReason.COMPLETED)
    """

    def __init__(
        self,
        config: LifecycleManagerConfig | None = None,
        client: BeadsClient | None = None,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            config: Manager configuration
            client: Optional pre-configured BeadsClient
        """
        self._config = config or LifecycleManagerConfig()
        self._client = client
        self._tracker = TicketTracker()
        self._initialized = False

        # Ticket history for progress tracking
        self._ticket_history: dict[str, list[dict[str, Any]]] = {}

        # Component to ticket mapping
        self._component_tickets: dict[str, str] = {}

    @property
    def config(self) -> LifecycleManagerConfig:
        """Get manager configuration."""
        return self._config

    @property
    def tracker(self) -> TicketTracker:
        """Get the ticket tracker."""
        return self._tracker

    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the manager and verify Beads connection.

        Raises:
            BeadsError: If initialization fails
        """
        if self._initialized:
            return

        # Create client if not provided
        if self._client is None:
            client_config = BeadsClientConfig(
                directory=self._config.beads_directory,
                labels=self._config.default_labels,
            )
            self._client = BeadsClient(client_config)

        # Initialize client
        if not self._client.is_initialized:
            await self._client.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Close the manager and release resources."""
        if self._client:
            await self._client.close()
        self._initialized = False

    async def create_documentation_ticket(
        self,
        component: "SourceComponent",
        priority: int = 1,
        labels: list[str] | None = None,
        parent_ticket_id: str | None = None,
    ) -> str:
        """Create a documentation ticket for a source component.

        Creates a Beads ticket to track the documentation process for
        a specific code component (function, class, method, etc.).

        Args:
            component: Source component to document
            priority: Ticket priority (0=highest)
            labels: Additional labels for the ticket
            parent_ticket_id: Parent ticket for subtask relationship

        Returns:
            Created ticket ID (e.g., "twinscribe-abc")

        Raises:
            BeadsError: If ticket creation fails
        """
        self._ensure_initialized()

        # Build ticket title
        component_name = getattr(component, "name", str(component))
        component_type = getattr(component, "type", "component")
        file_path = getattr(component, "file_path", "")
        component_id = getattr(component, "component_id", component_name)

        title = f"[DOC] Document {component_type}: {component_name}"

        # Build description
        description = self._build_documentation_ticket_description(component)

        # Merge labels
        ticket_labels = list(self._config.default_labels)
        ticket_labels.append(f"type-{component_type}")
        if labels:
            ticket_labels.extend(labels)
        ticket_labels = list(set(ticket_labels))

        # Create the ticket
        request = CreateIssueRequest(
            title=title,
            description=description,
            priority=priority,
            labels=ticket_labels,
            parent_id=parent_ticket_id,
        )

        issue = await self._client.create_issue(request)

        # Track the ticket
        self._tracker.track(
            ticket_key=issue.id,
            ticket_type=TicketType.REBUILD,  # Using rebuild for documentation tasks
            component_id=component_id,
            metadata={
                "component_name": component_name,
                "component_type": component_type,
                "file_path": file_path,
                "created_for": "documentation",
            },
        )

        # Store mapping
        self._component_tickets[component_id] = issue.id

        # Initialize history
        self._ticket_history[issue.id] = [
            {
                "action": "created",
                "status": DocumentationTicketStatus.OPEN.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]

        # Sync if configured
        if self._config.auto_sync:
            await self._safe_sync()

        return issue.id

    async def update_ticket_status(
        self,
        ticket_id: str,
        status: str | DocumentationTicketStatus,
        message: str | None = None,
    ) -> None:
        """Update the status of a documentation ticket.

        Args:
            ticket_id: Beads ticket ID
            status: New status value
            message: Optional status message

        Raises:
            NotFoundError: If ticket doesn't exist
            BeadsError: If update fails
        """
        self._ensure_initialized()

        # Normalize status
        if isinstance(status, DocumentationTicketStatus):
            status_value = status.value
        else:
            status_value = status

        # Map documentation status to Beads status
        beads_status = self._map_to_beads_status(status_value)

        # Update in Beads
        await self._client.update_issue(ticket_id, status=beads_status)

        # Update tracker
        tracked = self._tracker.get(ticket_id)
        if tracked:
            if status_value == DocumentationTicketStatus.IN_PROGRESS.value:
                tracked.status = TicketStatus.IN_PROGRESS
            elif status_value == DocumentationTicketStatus.RESOLVED.value:
                tracked.status = TicketStatus.RESOLVED
            elif status_value == DocumentationTicketStatus.CLOSED.value:
                tracked.status = TicketStatus.APPLIED
            tracked.updated_at = datetime.utcnow()

        # Record in history
        self._add_to_history(
            ticket_id,
            {
                "action": "status_update",
                "status": status_value,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        if self._config.auto_sync:
            await self._safe_sync()

    async def record_validation_result(
        self,
        ticket_id: str,
        result: "ValidationResult",
    ) -> None:
        """Record validation results on a ticket.

        Records the outcome of validation (from validator agents A2/B2)
        as an update to the documentation ticket.

        Args:
            ticket_id: Beads ticket ID
            result: ValidationResult from validator agent

        Raises:
            NotFoundError: If ticket doesn't exist
            BeadsError: If update fails
        """
        self._ensure_initialized()

        # Build validation summary
        summary = ValidationSummary(
            component_id=result.component_id,
            status=result.validation_result.value,
            completeness_score=result.completeness.score,
            call_graph_accuracy=result.call_graph_accuracy.score,
            corrections_count=result.total_corrections,
            errors=[
                *result.completeness.missing_elements,
                *result.call_graph_accuracy.false_callees,
            ],
        )

        # Build update message
        message = self._build_validation_message(summary)

        # Record in history with full details
        self._add_to_history(
            ticket_id,
            {
                "action": "validation_recorded",
                "validation": summary.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Update ticket status based on validation result
        if result.validation_result.value == "pass":
            await self.update_ticket_status(
                ticket_id,
                DocumentationTicketStatus.VALIDATING,
                message=message,
            )
        elif result.validation_result.value == "fail":
            await self.update_ticket_status(
                ticket_id,
                DocumentationTicketStatus.AWAITING_REVIEW,
                message=message,
            )

    async def record_convergence_metrics(
        self,
        ticket_id: str,
        metrics: ConvergenceMetrics,
    ) -> None:
        """Record convergence metrics on a ticket.

        Records progress toward convergence between documentation streams.

        Args:
            ticket_id: Beads ticket ID
            metrics: Convergence metrics to record

        Raises:
            NotFoundError: If ticket doesn't exist
        """
        self._ensure_initialized()

        # Record in history
        self._add_to_history(
            ticket_id,
            {
                "action": "convergence_metrics",
                "metrics": metrics.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Update tracker metadata
        tracked = self._tracker.get(ticket_id)
        if tracked:
            tracked.metadata["last_convergence"] = metrics.to_dict()
            tracked.updated_at = datetime.utcnow()

    async def link_dependencies(
        self,
        ticket_id: str,
        depends_on: list[str],
    ) -> None:
        """Link dependencies between tickets.

        Creates blocking relationships where the given ticket depends
        on the specified tickets being completed first.

        Args:
            ticket_id: Beads ticket ID that has dependencies
            depends_on: List of ticket IDs this ticket depends on

        Raises:
            NotFoundError: If ticket doesn't exist
            BeadsError: If linking fails
        """
        self._ensure_initialized()

        for dep_id in depends_on:
            try:
                await self._client.add_dependency(ticket_id, dep_id)
            except BeadsError as e:
                # Log but continue - some deps might not exist
                self._add_to_history(
                    ticket_id,
                    {
                        "action": "dependency_link_failed",
                        "depends_on": dep_id,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                continue

        # Record successful links in history
        self._add_to_history(
            ticket_id,
            {
                "action": "dependencies_linked",
                "depends_on": depends_on,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        if self._config.auto_sync:
            await self._safe_sync()

    async def close_ticket(
        self,
        ticket_id: str,
        reason: str | CloseReason,
        summary: str | None = None,
    ) -> None:
        """Close a documentation ticket.

        Args:
            ticket_id: Beads ticket ID
            reason: Reason for closing
            summary: Optional summary of work completed

        Raises:
            NotFoundError: If ticket doesn't exist
            BeadsError: If close fails
        """
        self._ensure_initialized()

        # Normalize reason
        if isinstance(reason, CloseReason):
            reason_value = reason.value
        else:
            reason_value = reason

        # Close in Beads
        await self._client.close_issue(ticket_id)

        # Update tracker
        tracked = self._tracker.get(ticket_id)
        if tracked:
            tracked.status = TicketStatus.APPLIED
            tracked.updated_at = datetime.utcnow()
            tracked.metadata["close_reason"] = reason_value
            if summary:
                tracked.metadata["close_summary"] = summary

        # Record in history
        self._add_to_history(
            ticket_id,
            {
                "action": "closed",
                "reason": reason_value,
                "summary": summary,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        if self._config.auto_sync:
            await self._safe_sync()

    async def get_ticket_history(
        self,
        ticket_id: str,
    ) -> list[dict[str, Any]]:
        """Get the history of actions on a ticket.

        Args:
            ticket_id: Beads ticket ID

        Returns:
            List of history entries
        """
        return self._ticket_history.get(ticket_id, [])

    async def get_component_ticket(
        self,
        component_id: str,
    ) -> str | None:
        """Get the ticket ID for a component.

        Args:
            component_id: Component identifier

        Returns:
            Ticket ID or None if not found
        """
        return self._component_tickets.get(component_id)

    async def get_open_documentation_tickets(self) -> list[TrackedTicket]:
        """Get all open documentation tickets.

        Returns:
            List of tracked tickets that are still open
        """
        return [
            t
            for t in self._tracker._tickets.values()
            if t.status in (TicketStatus.PENDING, TicketStatus.IN_PROGRESS)
            and t.metadata.get("created_for") == "documentation"
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get lifecycle manager statistics.

        Returns:
            Dictionary of statistics
        """
        tracker_stats = self._tracker.get_statistics()

        doc_tickets = [
            t
            for t in self._tracker._tickets.values()
            if t.metadata.get("created_for") == "documentation"
        ]

        return {
            "total_documentation_tickets": len(doc_tickets),
            "tracker": tracker_stats,
            "components_with_tickets": len(self._component_tickets),
            "initialized": self._initialized,
        }

    def _ensure_initialized(self) -> None:
        """Ensure manager is initialized.

        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("BeadsLifecycleManager not initialized. Call initialize() first.")

    def _build_documentation_ticket_description(
        self,
        component: "SourceComponent",
    ) -> str:
        """Build ticket description for documentation task.

        Args:
            component: Source component

        Returns:
            Formatted description
        """
        component_name = getattr(component, "name", str(component))
        component_type = getattr(component, "type", "component")
        file_path = getattr(component, "file_path", "unknown")
        signature = getattr(component, "signature", None)
        existing_docstring = getattr(component, "existing_docstring", None)

        lines = [
            "## Documentation Task",
            "",
            f"**Component:** `{component_name}`",
            f"**Type:** {component_type}",
            f"**Location:** `{file_path}`",
            "",
        ]

        if signature:
            lines.extend(
                [
                    "### Signature",
                    "```python",
                    signature,
                    "```",
                    "",
                ]
            )

        if existing_docstring:
            lines.extend(
                [
                    "### Existing Documentation",
                    "```",
                    existing_docstring,
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "## Process",
                "",
                "- [ ] Stream A documentation generated",
                "- [ ] Stream B documentation generated",
                "- [ ] Validation completed",
                "- [ ] Comparison completed",
                "- [ ] Convergence achieved",
                "",
                "## Notes",
                "",
                "_Documentation generated by TwinScribe dual-stream system._",
            ]
        )

        return "\n".join(lines)

    def _build_validation_message(
        self,
        summary: ValidationSummary,
    ) -> str:
        """Build a message from validation summary.

        Args:
            summary: Validation summary

        Returns:
            Formatted message
        """
        lines = [
            f"**Validation Result:** {summary.status}",
            f"- Completeness: {summary.completeness_score:.2%}",
            f"- Call Graph Accuracy: {summary.call_graph_accuracy:.2%}",
            f"- Corrections Applied: {summary.corrections_count}",
        ]

        if summary.errors:
            lines.append(f"- Errors: {len(summary.errors)}")
            for error in summary.errors[:5]:  # Limit to first 5
                lines.append(f"  - {error}")

        return "\n".join(lines)

    def _map_to_beads_status(self, status: str) -> str:
        """Map documentation status to Beads status.

        Args:
            status: Documentation ticket status

        Returns:
            Beads-compatible status string
        """
        mapping = {
            DocumentationTicketStatus.OPEN.value: "open",
            DocumentationTicketStatus.IN_PROGRESS.value: "in_progress",
            DocumentationTicketStatus.DOCUMENTING.value: "in_progress",
            DocumentationTicketStatus.VALIDATING.value: "in_progress",
            DocumentationTicketStatus.COMPARING.value: "in_progress",
            DocumentationTicketStatus.AWAITING_REVIEW.value: "open",
            DocumentationTicketStatus.RESOLVED.value: "resolved",
            DocumentationTicketStatus.CLOSED.value: "closed",
        }
        return mapping.get(status, "open")

    def _add_to_history(
        self,
        ticket_id: str,
        entry: dict[str, Any],
    ) -> None:
        """Add an entry to ticket history.

        Args:
            ticket_id: Ticket ID
            entry: History entry to add
        """
        if ticket_id not in self._ticket_history:
            self._ticket_history[ticket_id] = []
        self._ticket_history[ticket_id].append(entry)

    async def _safe_sync(self) -> None:
        """Sync with git, ignoring errors."""
        try:
            await self._client.sync()
        except BeadsError:
            pass  # Ignore sync errors


# Type aliases for compatibility
SourceComponent = Any  # Would be twinscribe.models.Component
ValidationResult = Any  # Would be twinscribe.models.ValidationResult
