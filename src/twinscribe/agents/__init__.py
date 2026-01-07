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
from twinscribe.agents.comparator import (
    ComparatorAgent,
    ComparatorConfig,
    ComparatorInput,
)
from twinscribe.agents.comparator_impl import (
    ConcreteComparatorAgent,
    create_comparator_agent,
)
from twinscribe.agents.documenter import (
    DocumenterAgent,
    DocumenterConfig,
    DocumenterInput,
)
from twinscribe.agents.stream import (
    ComponentProcessingResult,
    ConcreteDocumentationStream,
    ConcreteDocumenterAgent,
    ConcreteValidatorAgent,
    DocumentationStream,
    StreamConfig,
    StreamProgressCallback,
    StreamResult,
)
from twinscribe.agents.validator import (
    ValidatorAgent,
    ValidatorConfig,
    ValidatorInput,
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
    "ConcreteDocumenterAgent",
    # Validator
    "ValidatorAgent",
    "ValidatorInput",
    "ValidatorConfig",
    "ConcreteValidatorAgent",
    # Comparator
    "ComparatorAgent",
    "ComparatorInput",
    "ComparatorConfig",
    "ConcreteComparatorAgent",
    "create_comparator_agent",
    # Stream
    "DocumentationStream",
    "ConcreteDocumentationStream",
    "StreamConfig",
    "StreamResult",
    "StreamProgressCallback",
    "ComponentProcessingResult",
]
