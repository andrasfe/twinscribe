"""
TwinScribe: Dual-Stream Code Documentation System with Tiered Model Architecture.

A multi-agent system for generating accurate code documentation with call graph
linkages. Two independent agent streams document and validate code in parallel,
with discrepancies resolved through a premium-tier arbitrator agent.

Key Features:
- Dual documentation streams for increased accuracy
- Tiered model architecture for cost optimization (90% cost reduction)
- Static analysis anchoring for call graph validation
- Beads integration for human-in-the-loop resolution

Example:
    from twinscribe import DualStreamOrchestrator

    orchestrator = DualStreamOrchestrator(
        codebase_path="/path/to/codebase",
        language="python",
    )
    result = await orchestrator.run()
"""

from twinscribe.version import __version__

__all__ = [
    "__version__",
]
