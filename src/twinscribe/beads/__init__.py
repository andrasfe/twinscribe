"""
Dual-Stream Documentation System - Beads Integration

This module handles integration with Beads (git-backed issue tracker) for:

- Discrepancy issues: Created when streams disagree and require human review
- Rebuild issues: Final output for rebuilding documented components
- Lifecycle management: Creating, monitoring, and applying issue resolutions

Key Components:
- BeadsLifecycleManager: Manages issue creation and resolution monitoring
- BeadsClient: CLI wrapper for Beads (bd) commands
- IssueTracker: Maps issues to discrepancies
- IssueTemplateEngine: Renders issue content from templates

The Beads integration enables human-in-the-loop resolution for edge cases
that cannot be automatically resolved by ground truth or model consensus.

See https://github.com/steveyegge/beads for Beads documentation.
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
