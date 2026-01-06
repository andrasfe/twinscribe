"""
Dual-Stream Documentation System - Agent Interfaces

This module provides abstract base classes and protocols for all agents
in the documentation system:

- DocumenterAgent: Generates comprehensive documentation with call graph linkages
  - Stream A uses Claude Sonnet 4.5
  - Stream B uses GPT-4o

- ValidatorAgent: Verifies documentation completeness and call graph accuracy
  - Stream A uses Claude Haiku 4.5
  - Stream B uses GPT-4o-mini

- ComparatorAgent: Compares outputs, resolves discrepancies, generates Beads tickets
  - Uses Claude Opus 4.5 for complex judgment tasks

Agents follow a tiered model architecture for cost optimization.
"""

from twinscribe.agents.base import (
    AgentConfig,
    AgentMetrics,
    BaseAgent,
)
from twinscribe.agents.documenter import (
    DocumenterAgent,
    DocumenterInput,
    DocumenterConfig,
)
from twinscribe.agents.validator import (
    ValidatorAgent,
    ValidatorInput,
    ValidatorConfig,
)
from twinscribe.agents.comparator import (
    ComparatorAgent,
    ComparatorInput,
    ComparatorConfig,
)
from twinscribe.agents.stream import (
    DocumentationStream,
    StreamConfig,
    StreamResult,
)

__all__ = [
    # Base
    "AgentConfig",
    "AgentMetrics",
    "BaseAgent",
    # Documenter
    "DocumenterAgent",
    "DocumenterInput",
    "DocumenterConfig",
    # Validator
    "ValidatorAgent",
    "ValidatorInput",
    "ValidatorConfig",
    # Comparator
    "ComparatorAgent",
    "ComparatorInput",
    "ComparatorConfig",
    # Stream
    "DocumentationStream",
    "StreamConfig",
    "StreamResult",
]
