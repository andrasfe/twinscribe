"""
Dual-Stream Documentation System - Core Data Models

This module provides Pydantic models for the entire documentation system.
All models support JSON serialization and have comprehensive validation.
"""

from twinscribe.models.base import (
    CallType,
    ComponentType,
    DiscrepancyType,
    ResolutionAction,
    ValidationStatus,
)
from twinscribe.models.beads import (
    BeadsTicketPriority,
    BeadsTicketType,
    DiscrepancyTicket,
    RebuildTicket,
)
from twinscribe.models.call_graph import (
    CallEdge,
    CallGraph,
    CallGraphDiff,
)
from twinscribe.models.comparison import (
    ComparatorMetadata,
    ComparisonResult,
    ComparisonSummary,
    ConvergenceStatus,
    Discrepancy,
)
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ComponentLocation,
    ExceptionDoc,
    ParameterDoc,
    ReturnDoc,
)
from twinscribe.models.convergence import (
    ConvergenceCriteria,
    ConvergenceHistoryEntry,
    ConvergenceReport,
)
from twinscribe.models.documentation import (
    CalleeRef,
    CallerRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
)
from twinscribe.models.output import (
    ComponentFinalDoc,
    CostBreakdown,
    DocumentationPackage,
    RunMetrics,
)
from twinscribe.models.validation import (
    CallGraphAccuracy,
    CompletenessCheck,
    CorrectionApplied,
    ValidationResult,
    ValidatorMetadata,
)

__all__ = [
    # Base enums
    "ComponentType",
    "CallType",
    "DiscrepancyType",
    "ValidationStatus",
    "ResolutionAction",
    # Component models
    "Component",
    "ComponentLocation",
    "ParameterDoc",
    "ReturnDoc",
    "ExceptionDoc",
    "ComponentDocumentation",
    # Call graph models
    "CallEdge",
    "CallGraph",
    "CallGraphDiff",
    # Documentation output models
    "DocumenterMetadata",
    "DocumentationOutput",
    "CalleeRef",
    "CallerRef",
    "CallGraphSection",
    # Validation models
    "CompletenessCheck",
    "CallGraphAccuracy",
    "CorrectionApplied",
    "ValidatorMetadata",
    "ValidationResult",
    # Comparison models
    "Discrepancy",
    "ConvergenceStatus",
    "ComparisonSummary",
    "ComparatorMetadata",
    "ComparisonResult",
    # Convergence models
    "ConvergenceCriteria",
    "ConvergenceHistoryEntry",
    "ConvergenceReport",
    # Output models
    "ComponentFinalDoc",
    "DocumentationPackage",
    "RunMetrics",
    "CostBreakdown",
    # Beads models
    "BeadsTicketPriority",
    "BeadsTicketType",
    "DiscrepancyTicket",
    "RebuildTicket",
]
