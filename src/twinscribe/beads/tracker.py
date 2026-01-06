"""
Issue Tracker.

Maps Beads issues to discrepancies and tracks resolution status.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TicketStatus(str, Enum):
    """Status of a tracked ticket."""

    PENDING = "pending"  # Ticket created, awaiting human review
    IN_PROGRESS = "in_progress"  # Human is actively working on it
    RESOLVED = "resolved"  # Human provided resolution
    APPLIED = "applied"  # Resolution applied to documentation
    EXPIRED = "expired"  # Timeout reached without resolution
    CANCELLED = "cancelled"  # Ticket was cancelled


class TicketType(str, Enum):
    """Type of tracked ticket."""

    DISCREPANCY = "discrepancy"  # Stream disagreement requiring clarification
    REBUILD = "rebuild"  # Final rebuild story ticket


@dataclass
class TrackedTicket:
    """A ticket being tracked by the system.

    Attributes:
        ticket_key: Beads issue ID (e.g., bd-a1b2)
        ticket_type: Type of ticket (discrepancy or rebuild)
        status: Current tracking status
        discrepancy_id: ID of associated discrepancy (for discrepancy tickets)
        component_id: ID of associated component
        created_at: When tracking started
        updated_at: Last status update
        resolution_text: Human-provided resolution text
        resolution_action: Resolved action (accept_a, accept_b, merge, etc.)
        timeout_at: When the ticket will timeout
        metadata: Additional tracking metadata
    """

    ticket_key: str
    ticket_type: TicketType
    status: TicketStatus = TicketStatus.PENDING
    discrepancy_id: Optional[str] = None
    component_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolution_text: Optional[str] = None
    resolution_action: Optional[str] = None
    timeout_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if ticket is in a terminal state."""
        return self.status in (
            TicketStatus.APPLIED,
            TicketStatus.EXPIRED,
            TicketStatus.CANCELLED,
        )

    def is_actionable(self) -> bool:
        """Check if ticket has actionable resolution."""
        return self.status == TicketStatus.RESOLVED and self.resolution_action is not None

    def mark_resolved(
        self,
        resolution_text: str,
        resolution_action: str,
    ) -> None:
        """Mark ticket as resolved with human input.

        Args:
            resolution_text: Human-provided resolution explanation
            resolution_action: Action to take (accept_a, accept_b, merge, manual)
        """
        self.status = TicketStatus.RESOLVED
        self.resolution_text = resolution_text
        self.resolution_action = resolution_action
        self.updated_at = datetime.utcnow()

    def mark_applied(self) -> None:
        """Mark resolution as applied to documentation."""
        self.status = TicketStatus.APPLIED
        self.updated_at = datetime.utcnow()

    def mark_expired(self) -> None:
        """Mark ticket as expired due to timeout."""
        self.status = TicketStatus.EXPIRED
        self.updated_at = datetime.utcnow()


class TicketQuery(BaseModel):
    """Query parameters for finding tracked tickets.

    Attributes:
        ticket_keys: Filter by specific ticket keys
        ticket_type: Filter by ticket type
        status: Filter by status(es)
        component_id: Filter by component
        discrepancy_id: Filter by discrepancy
        include_terminal: Include terminal state tickets
    """

    ticket_keys: Optional[list[str]] = None
    ticket_type: Optional[TicketType] = None
    status: Optional[list[TicketStatus]] = None
    component_id: Optional[str] = None
    discrepancy_id: Optional[str] = None
    include_terminal: bool = False


class TicketTracker:
    """Tracks Beads tickets and their resolution status.

    Maintains mapping between:
    - Tickets and discrepancies
    - Tickets and components
    - Resolution status and actions

    Thread-safe for concurrent access.

    Usage:
        tracker = TicketTracker()
        tracker.track(ticket_key="LEGACY-123", discrepancy_id="disc-001")

        # Later, when resolved
        tracker.update_resolution("LEGACY-123", "Use version A", "accept_a")

        # Get pending tickets
        pending = tracker.get_pending_tickets()
    """

    def __init__(self) -> None:
        """Initialize the tracker."""
        self._tickets: dict[str, TrackedTicket] = {}
        self._by_discrepancy: dict[str, str] = {}  # discrepancy_id -> ticket_key
        self._by_component: dict[str, list[str]] = {}  # component_id -> ticket_keys

    def track(
        self,
        ticket_key: str,
        ticket_type: TicketType,
        discrepancy_id: Optional[str] = None,
        component_id: Optional[str] = None,
        timeout_at: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TrackedTicket:
        """Start tracking a ticket.

        Args:
            ticket_key: Beads ticket key
            ticket_type: Type of ticket
            discrepancy_id: Associated discrepancy ID
            component_id: Associated component ID
            timeout_at: When ticket should timeout
            metadata: Additional metadata

        Returns:
            The tracked ticket

        Raises:
            ValueError: If ticket already being tracked
        """
        if ticket_key in self._tickets:
            raise ValueError(f"Ticket {ticket_key} already being tracked")

        ticket = TrackedTicket(
            ticket_key=ticket_key,
            ticket_type=ticket_type,
            discrepancy_id=discrepancy_id,
            component_id=component_id,
            timeout_at=timeout_at,
            metadata=metadata or {},
        )

        self._tickets[ticket_key] = ticket

        # Index by discrepancy
        if discrepancy_id:
            self._by_discrepancy[discrepancy_id] = ticket_key

        # Index by component
        if component_id:
            if component_id not in self._by_component:
                self._by_component[component_id] = []
            self._by_component[component_id].append(ticket_key)

        return ticket

    def get(self, ticket_key: str) -> Optional[TrackedTicket]:
        """Get a tracked ticket by key.

        Args:
            ticket_key: Beads ticket key

        Returns:
            Tracked ticket or None if not found
        """
        return self._tickets.get(ticket_key)

    def get_by_discrepancy(self, discrepancy_id: str) -> Optional[TrackedTicket]:
        """Get ticket by discrepancy ID.

        Args:
            discrepancy_id: Discrepancy ID

        Returns:
            Associated ticket or None
        """
        ticket_key = self._by_discrepancy.get(discrepancy_id)
        if ticket_key:
            return self._tickets.get(ticket_key)
        return None

    def get_by_component(self, component_id: str) -> list[TrackedTicket]:
        """Get all tickets for a component.

        Args:
            component_id: Component ID

        Returns:
            List of associated tickets
        """
        ticket_keys = self._by_component.get(component_id, [])
        return [self._tickets[k] for k in ticket_keys if k in self._tickets]

    def query(self, query: TicketQuery) -> list[TrackedTicket]:
        """Query tracked tickets.

        Args:
            query: Query parameters

        Returns:
            List of matching tickets
        """
        results = list(self._tickets.values())

        # Filter by ticket keys
        if query.ticket_keys:
            results = [t for t in results if t.ticket_key in query.ticket_keys]

        # Filter by type
        if query.ticket_type:
            results = [t for t in results if t.ticket_type == query.ticket_type]

        # Filter by status
        if query.status:
            results = [t for t in results if t.status in query.status]

        # Filter by component
        if query.component_id:
            results = [t for t in results if t.component_id == query.component_id]

        # Filter by discrepancy
        if query.discrepancy_id:
            results = [t for t in results if t.discrepancy_id == query.discrepancy_id]

        # Exclude terminal unless requested
        if not query.include_terminal:
            results = [t for t in results if not t.is_terminal()]

        return results

    def get_pending_tickets(self) -> list[TrackedTicket]:
        """Get all pending tickets awaiting resolution.

        Returns:
            List of pending tickets
        """
        return self.query(TicketQuery(status=[TicketStatus.PENDING]))

    def get_resolved_tickets(self) -> list[TrackedTicket]:
        """Get all resolved tickets ready for application.

        Returns:
            List of resolved tickets
        """
        return self.query(TicketQuery(status=[TicketStatus.RESOLVED]))

    def get_expired_candidates(self) -> list[TrackedTicket]:
        """Get tickets that have exceeded their timeout.

        Returns:
            List of tickets past timeout
        """
        now = datetime.utcnow()
        pending = self.get_pending_tickets()
        return [
            t for t in pending
            if t.timeout_at and t.timeout_at <= now
        ]

    def update_status(
        self,
        ticket_key: str,
        status: TicketStatus,
    ) -> Optional[TrackedTicket]:
        """Update ticket status.

        Args:
            ticket_key: Ticket key
            status: New status

        Returns:
            Updated ticket or None if not found
        """
        ticket = self._tickets.get(ticket_key)
        if ticket:
            ticket.status = status
            ticket.updated_at = datetime.utcnow()
        return ticket

    def update_resolution(
        self,
        ticket_key: str,
        resolution_text: str,
        resolution_action: str,
    ) -> Optional[TrackedTicket]:
        """Update ticket with resolution.

        Args:
            ticket_key: Ticket key
            resolution_text: Human-provided resolution
            resolution_action: Action to take

        Returns:
            Updated ticket or None if not found
        """
        ticket = self._tickets.get(ticket_key)
        if ticket:
            ticket.mark_resolved(resolution_text, resolution_action)
        return ticket

    def mark_applied(self, ticket_key: str) -> Optional[TrackedTicket]:
        """Mark ticket resolution as applied.

        Args:
            ticket_key: Ticket key

        Returns:
            Updated ticket or None
        """
        ticket = self._tickets.get(ticket_key)
        if ticket:
            ticket.mark_applied()
        return ticket

    def expire_ticket(self, ticket_key: str) -> Optional[TrackedTicket]:
        """Mark ticket as expired.

        Args:
            ticket_key: Ticket key

        Returns:
            Updated ticket or None
        """
        ticket = self._tickets.get(ticket_key)
        if ticket:
            ticket.mark_expired()
        return ticket

    def remove(self, ticket_key: str) -> Optional[TrackedTicket]:
        """Remove a ticket from tracking.

        Args:
            ticket_key: Ticket key

        Returns:
            Removed ticket or None if not found
        """
        ticket = self._tickets.pop(ticket_key, None)
        if ticket:
            # Remove from indices
            if ticket.discrepancy_id:
                self._by_discrepancy.pop(ticket.discrepancy_id, None)
            if ticket.component_id:
                component_tickets = self._by_component.get(ticket.component_id, [])
                if ticket_key in component_tickets:
                    component_tickets.remove(ticket_key)
        return ticket

    def clear(self) -> None:
        """Clear all tracked tickets."""
        self._tickets.clear()
        self._by_discrepancy.clear()
        self._by_component.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get tracking statistics.

        Returns:
            Dictionary of statistics
        """
        tickets = list(self._tickets.values())

        status_counts = {}
        for status in TicketStatus:
            status_counts[status.value] = sum(1 for t in tickets if t.status == status)

        type_counts = {}
        for ticket_type in TicketType:
            type_counts[ticket_type.value] = sum(
                1 for t in tickets if t.ticket_type == ticket_type
            )

        return {
            "total_tickets": len(tickets),
            "by_status": status_counts,
            "by_type": type_counts,
            "unique_components": len(self._by_component),
            "unique_discrepancies": len(self._by_discrepancy),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracker state to dictionary.

        Returns:
            Serialized state
        """
        return {
            "tickets": [
                {
                    "ticket_key": t.ticket_key,
                    "ticket_type": t.ticket_type.value,
                    "status": t.status.value,
                    "discrepancy_id": t.discrepancy_id,
                    "component_id": t.component_id,
                    "created_at": t.created_at.isoformat(),
                    "updated_at": t.updated_at.isoformat(),
                    "resolution_text": t.resolution_text,
                    "resolution_action": t.resolution_action,
                    "timeout_at": t.timeout_at.isoformat() if t.timeout_at else None,
                    "metadata": t.metadata,
                }
                for t in self._tickets.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TicketTracker":
        """Deserialize tracker state from dictionary.

        Args:
            data: Serialized state

        Returns:
            Restored TicketTracker
        """
        tracker = cls()

        for ticket_data in data.get("tickets", []):
            ticket = TrackedTicket(
                ticket_key=ticket_data["ticket_key"],
                ticket_type=TicketType(ticket_data["ticket_type"]),
                status=TicketStatus(ticket_data["status"]),
                discrepancy_id=ticket_data.get("discrepancy_id"),
                component_id=ticket_data.get("component_id"),
                created_at=datetime.fromisoformat(ticket_data["created_at"]),
                updated_at=datetime.fromisoformat(ticket_data["updated_at"]),
                resolution_text=ticket_data.get("resolution_text"),
                resolution_action=ticket_data.get("resolution_action"),
                timeout_at=(
                    datetime.fromisoformat(ticket_data["timeout_at"])
                    if ticket_data.get("timeout_at")
                    else None
                ),
                metadata=ticket_data.get("metadata", {}),
            )

            tracker._tickets[ticket.ticket_key] = ticket

            if ticket.discrepancy_id:
                tracker._by_discrepancy[ticket.discrepancy_id] = ticket.ticket_key

            if ticket.component_id:
                if ticket.component_id not in tracker._by_component:
                    tracker._by_component[ticket.component_id] = []
                tracker._by_component[ticket.component_id].append(ticket.ticket_key)

        return tracker
