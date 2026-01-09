"""
Dual-Stream Documentation System - Beads Integration

This module handles integration with Beads (git-backed issue tracker) for:

- Discrepancy issues: Created when streams disagree and require human review
- Rebuild issues: Final output for rebuilding documented components
- Lifecycle management: Creating, monitoring, and applying issue resolutions
- Documentation task tracking: Creating and updating tickets through workflow

Key Components:
- BeadsLifecycleManager: Manages issue creation and resolution monitoring
- BeadsClient: CLI wrapper for Beads (bd) commands
- IssueTracker: Maps issues to discrepancies
- IssueTemplateEngine: Renders issue content from templates
- Lifecycle classes: Documentation ticket lifecycle management

The Beads integration enables human-in-the-loop resolution for edge cases
that cannot be automatically resolved by ground truth or model consensus.

See https://github.com/steveyegge/beads for Beads documentation.
"""

from twinscribe.beads.client import (
    BeadsClient,
    BeadsClientConfig,
    BeadsError,
    BeadsIssue,
    CreateIssueRequest,
    NotFoundError,
)
from twinscribe.beads.lifecycle import (
    BeadsLifecycleManager as DocumentationLifecycleManager,
)
from twinscribe.beads.lifecycle import (
    CloseReason,
    ConvergenceMetrics,
    DocumentationTicketStatus,
    LifecycleManagerConfig,
    ValidationSummary,
)
from twinscribe.beads.manager import (
    BeadsLifecycleManager,
    ResolutionAction,
    ResolutionResult,
    TicketResolution,
)
from twinscribe.beads.templates import (
    DiscrepancyTemplateData,
    DivergentComponentTemplateData,
    RebuildTemplateData,
    TicketTemplateEngine,
)
from twinscribe.beads.tracker import (
    TicketStatus,
    TicketTracker,
    TicketType,
    TrackedTicket,
)

__all__ = [
    # Manager (discrepancy/rebuild focused)
    "BeadsLifecycleManager",
    "ResolutionAction",
    "TicketResolution",
    "ResolutionResult",
    # Documentation Lifecycle Manager
    "DocumentationLifecycleManager",
    "DocumentationTicketStatus",
    "CloseReason",
    "ConvergenceMetrics",
    "ValidationSummary",
    "LifecycleManagerConfig",
    # Client
    "BeadsClient",
    "BeadsClientConfig",
    "BeadsIssue",
    "CreateIssueRequest",
    "BeadsError",
    "NotFoundError",
    # Tracker
    "TicketTracker",
    "TrackedTicket",
    "TicketStatus",
    "TicketType",
    # Templates
    "TicketTemplateEngine",
    "DiscrepancyTemplateData",
    "DivergentComponentTemplateData",
    "RebuildTemplateData",
]
