"""
Dual-Stream Documentation System - Beads Integration

This module handles integration with issue tracking systems (Jira/Beads) for:

- Discrepancy tickets: Created when streams disagree and require human review
- Rebuild tickets: Final output for rebuilding documented components
- Lifecycle management: Creating, monitoring, and applying ticket resolutions

Key Components:
- BeadsLifecycleManager: Manages ticket creation and resolution monitoring
- BeadsClient: Low-level API client for Beads/Jira
- TicketTracker: Maps tickets to discrepancies
- TicketTemplateEngine: Renders ticket content from templates

The Beads integration enables human-in-the-loop resolution for edge cases
that cannot be automatically resolved by ground truth or model consensus.
"""

from twinscribe.beads.manager import (
    BeadsLifecycleManager,
    TicketResolution,
    ResolutionResult,
)
from twinscribe.beads.client import (
    BeadsClient,
    BeadsClientConfig,
)
from twinscribe.beads.tracker import (
    TicketTracker,
    TrackedTicket,
    TicketStatus,
)
from twinscribe.beads.templates import (
    TicketTemplateEngine,
    DiscrepancyTemplateData,
    RebuildTemplateData,
)

__all__ = [
    # Manager
    "BeadsLifecycleManager",
    "TicketResolution",
    "ResolutionResult",
    # Client
    "BeadsClient",
    "BeadsClientConfig",
    # Tracker
    "TicketTracker",
    "TrackedTicket",
    "TicketStatus",
    # Templates
    "TicketTemplateEngine",
    "DiscrepancyTemplateData",
    "RebuildTemplateData",
]
