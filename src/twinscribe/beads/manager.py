"""
Beads Lifecycle Manager.

Coordinates the full lifecycle of Beads tickets:
- Creating discrepancy tickets when streams disagree
- Monitoring ticket status for resolutions
- Applying resolutions to documentation
- Creating rebuild tickets from final documentation
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from twinscribe.beads.client import (
    BeadsClient,
    BeadsError,
    BeadsIssue,
    CreateIssueRequest,
)
from twinscribe.beads.templates import (
    DiscrepancyTemplateData,
    DivergentComponentTemplateData,
    RebuildTemplateData,
    ResolutionParser,
    TicketTemplateEngine,
)
from twinscribe.beads.tracker import (
    TicketStatus,
    TicketTracker,
    TicketType,
    TrackedTicket,
)


class ResolutionAction(str, Enum):
    """Actions that can be taken to resolve a discrepancy."""

    ACCEPT_A = "accept_a"  # Use Stream A's interpretation
    ACCEPT_B = "accept_b"  # Use Stream B's interpretation
    MERGE = "merge"  # Merge both interpretations
    MANUAL = "manual"  # Use manually provided content


@dataclass
class TicketResolution:
    """Resolution from a Beads ticket.

    Attributes:
        ticket_key: Beads ticket key
        action: Resolution action
        content: Resolution content (for merge/manual)
        resolved_by: Username who resolved
        resolved_at: Resolution timestamp
        comment_id: ID of the comment containing resolution
    """

    ticket_key: str
    action: ResolutionAction
    content: str = ""
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    comment_id: str | None = None


@dataclass
class ResolutionResult:
    """Result of applying a resolution.

    Attributes:
        success: Whether application succeeded
        ticket_key: Associated ticket key
        discrepancy_id: Associated discrepancy ID
        applied_value: The value that was applied
        error: Error message if failed
    """

    success: bool
    ticket_key: str
    discrepancy_id: str | None = None
    applied_value: str | None = None
    error: str | None = None


class ManagerConfig(BaseModel):
    """Configuration for BeadsLifecycleManager.

    Attributes:
        project: Project key for discrepancy tickets
        rebuild_project: Project key for rebuild tickets
        poll_interval_seconds: How often to poll for resolutions
        timeout_hours: Hours before a ticket times out
        auto_create_tickets: Whether to auto-create tickets
        default_labels: Default labels for tickets
        max_concurrent_polls: Max concurrent ticket polls
    """

    project: str = Field(default="LEGACY_DOC")
    rebuild_project: str = Field(default="REBUILD")
    poll_interval_seconds: int = Field(default=60, ge=10)
    timeout_hours: int = Field(default=48, ge=1)
    auto_create_tickets: bool = Field(default=True)
    default_labels: list[str] = Field(default_factory=lambda: ["ai-documentation"])
    max_concurrent_polls: int = Field(default=10, ge=1)


# Type alias for resolution callback
ResolutionCallback = Callable[[TicketResolution], None]


class BeadsLifecycleManager:
    """Manages the lifecycle of Beads tickets.

    Responsibilities:
    - Create discrepancy tickets when streams disagree
    - Monitor tickets for human resolution
    - Parse and apply resolutions
    - Handle timeouts and escalations
    - Create rebuild tickets from final documentation

    Usage:
        manager = BeadsLifecycleManager(client, config)
        await manager.initialize()

        # Create discrepancy ticket
        ticket = await manager.create_discrepancy_ticket(discrepancy_data)

        # Start monitoring for resolution
        await manager.start_monitoring()

        # Or wait for specific ticket
        resolution = await manager.wait_for_resolution(ticket.ticket_key)
    """

    def __init__(
        self,
        client: BeadsClient,
        config: ManagerConfig | None = None,
    ) -> None:
        """Initialize the lifecycle manager.

        Args:
            client: Beads API client
            config: Manager configuration
        """
        self._client = client
        self._config = config or ManagerConfig()
        self._tracker = TicketTracker()
        self._template_engine = TicketTemplateEngine()
        self._resolution_parser = ResolutionParser()
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._resolution_callbacks: list[ResolutionCallback] = []
        self._initialized = False

    @property
    def client(self) -> BeadsClient:
        """Get the Beads client."""
        return self._client

    @property
    def tracker(self) -> TicketTracker:
        """Get the ticket tracker."""
        return self._tracker

    @property
    def config(self) -> ManagerConfig:
        """Get the manager configuration."""
        return self._config

    @property
    def is_monitoring(self) -> bool:
        """Check if actively monitoring tickets."""
        return self._monitoring

    async def initialize(self) -> None:
        """Initialize the manager and verify connection.

        Raises:
            BeadsError: If connection fails
        """
        if not self._client.is_initialized:
            await self._client.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Close the manager and stop monitoring."""
        await self.stop_monitoring()
        self._initialized = False

    def on_resolution(self, callback: ResolutionCallback) -> None:
        """Register a callback for when tickets are resolved.

        Args:
            callback: Function to call with resolution
        """
        self._resolution_callbacks.append(callback)

    async def create_discrepancy_ticket(
        self,
        data: DiscrepancyTemplateData,
        template_name: str = "default_discrepancy",
    ) -> TrackedTicket:
        """Create a discrepancy ticket.

        Args:
            data: Template data for the ticket
            template_name: Template to use for rendering

        Returns:
            Tracked ticket

        Raises:
            BeadsError: If ticket creation fails
        """
        # Render ticket content
        summary, description = self._template_engine.render_discrepancy(data, template_name)

        # Get labels and priority
        labels = self._template_engine.get_labels(data, template_name)
        labels.extend(self._config.default_labels)
        priority = self._template_engine.get_priority(data, template_name)

        # Create the ticket
        request = CreateIssueRequest(
            project=self._config.project,
            issue_type="Task",
            summary=summary,
            description=description,
            priority=priority,
            labels=list(set(labels)),
            custom_fields={
                "discrepancy_id": data.discrepancy_id,
                "component_name": data.component_name,
                "iteration": data.iteration,
            },
        )

        issue = await self._client.create_issue(request)

        # Calculate timeout
        timeout_at = datetime.utcnow() + timedelta(hours=self._config.timeout_hours)

        # Track the ticket
        tracked = self._tracker.track(
            ticket_key=issue.key,
            ticket_type=TicketType.DISCREPANCY,
            discrepancy_id=data.discrepancy_id,
            component_id=data.component_name,
            timeout_at=timeout_at,
            metadata={
                "iteration": data.iteration,
                "discrepancy_type": data.discrepancy_type,
                "file_path": data.file_path,
            },
        )

        return tracked

    async def create_rebuild_ticket(
        self,
        data: RebuildTemplateData,
        template_name: str = "default_rebuild",
    ) -> TrackedTicket:
        """Create a rebuild ticket.

        Args:
            data: Template data for the ticket
            template_name: Template to use for rendering

        Returns:
            Tracked ticket

        Raises:
            BeadsError: If ticket creation fails
        """
        # Render ticket content
        summary, description = self._template_engine.render_rebuild(data, template_name)

        # Get labels and priority
        labels = self._template_engine.get_labels(data, template_name)
        labels.extend(self._config.default_labels)

        # Determine priority based on rebuild priority
        if data.rebuild_priority <= 3:
            priority = "High"
        elif data.rebuild_priority <= 10:
            priority = "Medium"
        else:
            priority = "Low"

        # Create the ticket
        request = CreateIssueRequest(
            project=self._config.rebuild_project,
            issue_type="Story",
            summary=summary,
            description=description,
            priority=priority,
            labels=list(set(labels)),
            custom_fields={
                "component_name": data.component_name,
                "rebuild_priority": data.rebuild_priority,
                "complexity_score": data.complexity_score,
            },
        )

        issue = await self._client.create_issue(request)

        # Link to epic if provided
        if data.epic_key:
            await self._link_to_epic(issue.key, data.epic_key)

        # Track the ticket
        tracked = self._tracker.track(
            ticket_key=issue.key,
            ticket_type=TicketType.REBUILD,
            component_id=data.component_name,
            metadata={
                "rebuild_priority": data.rebuild_priority,
                "complexity_score": data.complexity_score,
                "file_path": data.file_path,
            },
        )

        return tracked

    async def create_divergent_component_ticket(
        self,
        data: DivergentComponentTemplateData,
        template_name: str = "default_divergent_component",
    ) -> TrackedTicket:
        """Create a ticket for a divergent component.

        Called when streams fail to converge on call graphs after max iterations.
        The ticket includes both stream outputs and iteration history for human review.

        Args:
            data: Template data for the divergent component
            template_name: Template to use for rendering

        Returns:
            Tracked ticket

        Raises:
            BeadsError: If ticket creation fails
        """
        # Render ticket content
        summary, description = self._template_engine.render_divergent_component(data, template_name)

        # Get labels and priority
        labels = self._template_engine.get_labels(data, template_name)
        labels.extend(self._config.default_labels)
        priority = self._template_engine.get_priority(data, template_name)

        # Create the ticket
        request = CreateIssueRequest(
            project=self._config.project,
            issue_type="Task",
            summary=summary,
            description=description,
            priority=priority,
            labels=list(set(labels)),
            custom_fields={
                "component_id": data.component_id,
                "component_name": data.component_name,
                "total_iterations": data.total_iterations,
                "final_agreement_rate": data.final_agreement_rate,
                "discrepancy_type": "call_graph_divergence",
            },
        )

        issue = await self._client.create_issue(request)

        # Calculate timeout
        timeout_at = datetime.utcnow() + timedelta(hours=self._config.timeout_hours)

        # Track the ticket
        tracked = self._tracker.track(
            ticket_key=issue.key,
            ticket_type=TicketType.DISCREPANCY,
            discrepancy_id=f"divergent_{data.component_id}",
            component_id=data.component_id,
            timeout_at=timeout_at,
            metadata={
                "total_iterations": data.total_iterations,
                "final_agreement_rate": data.final_agreement_rate,
                "file_path": data.file_path,
                "edges_only_in_a": len(data.edges_only_in_a),
                "edges_only_in_b": len(data.edges_only_in_b),
                "common_edges": len(data.common_edges),
            },
        )

        return tracked

    async def check_for_resolution(
        self,
        ticket_key: str,
    ) -> TicketResolution | None:
        """Check if a ticket has been resolved.

        Args:
            ticket_key: Beads ticket key

        Returns:
            Resolution if found, None otherwise
        """
        # Get ticket status
        issue = await self._client.get_issue(ticket_key)

        # Check if closed/resolved
        if issue.is_resolved:
            # Check for resolution in resolution field
            if issue.resolution:
                return await self._extract_resolution_from_issue(issue)

        # Check comments for resolution
        comments = await self._client.get_comments(ticket_key)

        for comment in reversed(comments):  # Most recent first
            if self._resolution_parser.is_resolution_comment(comment.body):
                result = self._resolution_parser.parse(comment.body)
                if result:
                    action, content = result
                    return TicketResolution(
                        ticket_key=ticket_key,
                        action=ResolutionAction(action),
                        content=content,
                        resolved_by=comment.author,
                        resolved_at=comment.created,
                        comment_id=comment.id,
                    )

        return None

    async def wait_for_resolution(
        self,
        ticket_key: str,
        timeout_seconds: int | None = None,
        poll_interval: int | None = None,
    ) -> TicketResolution | None:
        """Wait for a ticket to be resolved.

        Args:
            ticket_key: Beads ticket key
            timeout_seconds: Max seconds to wait (None = use config)
            poll_interval: Seconds between polls (None = use config)

        Returns:
            Resolution if found within timeout, None if timeout reached
        """
        timeout = timeout_seconds or (self._config.timeout_hours * 3600)
        interval = poll_interval or self._config.poll_interval_seconds

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=timeout)

        while datetime.utcnow() < end_time:
            resolution = await self.check_for_resolution(ticket_key)
            if resolution:
                # Update tracker
                tracked = self._tracker.get(ticket_key)
                if tracked:
                    tracked.mark_resolved(
                        resolution.content,
                        resolution.action.value,
                    )
                return resolution

            # Wait before next poll
            await asyncio.sleep(interval)

        # Timeout reached
        self._tracker.expire_ticket(ticket_key)
        return None

    async def start_monitoring(self) -> None:
        """Start background monitoring of all pending tickets."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitoring_loop(self) -> None:
        """Background loop that monitors pending tickets."""
        while self._monitoring:
            try:
                # Get pending tickets
                pending = self._tracker.get_pending_tickets()

                # Check for expired tickets
                expired = self._tracker.get_expired_candidates()
                for ticket in expired:
                    self._tracker.expire_ticket(ticket.ticket_key)

                # Check for resolutions (with concurrency limit)
                semaphore = asyncio.Semaphore(self._config.max_concurrent_polls)

                async def check_ticket(
                    ticket: TrackedTicket, sem: asyncio.Semaphore = semaphore
                ) -> None:
                    async with sem:
                        try:
                            resolution = await self.check_for_resolution(ticket.ticket_key)
                            if resolution:
                                ticket.mark_resolved(
                                    resolution.content,
                                    resolution.action.value,
                                )
                                # Notify callbacks
                                for callback in self._resolution_callbacks:
                                    try:
                                        callback(resolution)
                                    except Exception:
                                        pass  # Don't let callback errors stop monitoring
                        except BeadsError:
                            pass  # Ignore transient errors

                await asyncio.gather(
                    *[check_ticket(t) for t in pending if t not in expired],
                    return_exceptions=True,
                )

            except Exception:
                pass  # Don't let errors stop the monitoring loop

            # Wait before next poll cycle
            await asyncio.sleep(self._config.poll_interval_seconds)

    async def apply_resolution(
        self,
        resolution: TicketResolution,
        apply_func: Callable[[ResolutionAction, str], str],
    ) -> ResolutionResult:
        """Apply a resolution to documentation.

        Args:
            resolution: The resolution to apply
            apply_func: Function that applies the resolution and returns result

        Returns:
            Result of applying the resolution
        """
        tracked = self._tracker.get(resolution.ticket_key)
        if not tracked:
            return ResolutionResult(
                success=False,
                ticket_key=resolution.ticket_key,
                error="Ticket not found in tracker",
            )

        try:
            # Apply the resolution
            applied_value = apply_func(resolution.action, resolution.content)

            # Mark as applied
            tracked.mark_applied()

            # Add comment to ticket
            await self._client.add_comment(
                resolution.ticket_key,
                f"Resolution applied automatically.\nApplied value:\n{{code}}\n{applied_value}\n{{code}}",
            )

            # Transition ticket if possible
            try:
                await self._client.transition_issue(
                    resolution.ticket_key,
                    "Done",
                    resolution="Done",
                )
            except BeadsError:
                pass  # Transition might not be available

            return ResolutionResult(
                success=True,
                ticket_key=resolution.ticket_key,
                discrepancy_id=tracked.discrepancy_id,
                applied_value=applied_value,
            )

        except Exception as e:
            return ResolutionResult(
                success=False,
                ticket_key=resolution.ticket_key,
                discrepancy_id=tracked.discrepancy_id,
                error=str(e),
            )

    async def create_rebuild_epic(
        self,
        epic_name: str,
        components: list[RebuildTemplateData],
    ) -> str:
        """Create an epic for rebuild tickets.

        Args:
            epic_name: Name for the epic
            components: Components to include in rebuild

        Returns:
            Epic ticket key
        """
        # Create epic
        total_complexity = sum(c.complexity_score for c in components)
        description = f"""
h2. Rebuild Epic

This epic contains {len(components)} components to rebuild.

*Total Complexity Score:* {total_complexity:.2f}

h2. Components

{chr(10).join(f"* {c.component_name} (priority: {c.rebuild_priority})" for c in sorted(components, key=lambda x: x.rebuild_priority))}
""".strip()

        request = CreateIssueRequest(
            project=self._config.rebuild_project,
            issue_type="Epic",
            summary=epic_name,
            description=description,
            priority="High",
            labels=self._config.default_labels + ["rebuild-epic"],
        )

        epic = await self._client.create_issue(request)
        return epic.key

    async def _link_to_epic(self, issue_key: str, epic_key: str) -> None:
        """Link an issue to an epic.

        Args:
            issue_key: Issue to link
            epic_key: Epic to link to
        """
        # This is typically done via custom field update
        await self._client.update_issue(
            issue_key,
            {"customfield_10014": epic_key},  # Epic link field (varies by instance)
        )

    async def _extract_resolution_from_issue(
        self,
        issue: BeadsIssue,
    ) -> TicketResolution | None:
        """Extract resolution from issue fields.

        Args:
            issue: Beads issue

        Returns:
            Resolution if extractable
        """
        # Try to get resolution from custom fields or description
        resolution_field = issue.custom_fields.get("resolution_action")
        content_field = issue.custom_fields.get("resolution_content")

        if resolution_field:
            try:
                action = ResolutionAction(resolution_field.lower())
                return TicketResolution(
                    ticket_key=issue.key,
                    action=action,
                    content=content_field or "",
                    resolved_at=issue.updated,
                )
            except ValueError:
                pass

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dictionary of statistics
        """
        tracker_stats = self._tracker.get_statistics()

        return {
            "tracker": tracker_stats,
            "monitoring_active": self._monitoring,
            "config": {
                "project": self._config.project,
                "rebuild_project": self._config.rebuild_project,
                "timeout_hours": self._config.timeout_hours,
                "poll_interval_seconds": self._config.poll_interval_seconds,
            },
        }

    async def sync_from_beads(self, jql: str) -> int:
        """Sync tracker state from existing Beads tickets.

        Useful for recovering state after restart.

        Args:
            jql: JQL query to find relevant tickets

        Returns:
            Number of tickets synced
        """
        issues = await self._client.search_issues(jql)
        synced = 0

        for issue in issues:
            if issue.key in self._tracker._tickets:
                continue  # Already tracking

            # Determine ticket type from labels
            ticket_type = TicketType.REBUILD
            if "discrepancy" in issue.labels:
                ticket_type = TicketType.DISCREPANCY

            # Determine status
            status = TicketStatus.PENDING
            if issue.is_resolved:
                status = TicketStatus.RESOLVED

            # Extract metadata from custom fields
            discrepancy_id = issue.custom_fields.get("discrepancy_id")
            component_id = issue.custom_fields.get("component_name")

            # Track the ticket
            tracked = self._tracker.track(
                ticket_key=issue.key,
                ticket_type=ticket_type,
                discrepancy_id=discrepancy_id,
                component_id=component_id,
                metadata=issue.custom_fields,
            )
            tracked.status = status
            tracked.created_at = issue.created or datetime.utcnow()
            tracked.updated_at = issue.updated or datetime.utcnow()

            synced += 1

        return synced
