"""
Dual-Stream Documentation System - Core Data Models

This module provides Pydantic models for the entire documentation system.
All models support JSON serialization and have comprehensive validation.
"""

from twinscribe.models.base import (
    ComponentType,
    CallType,
    DiscrepancyType,
    ValidationStatus,
    ResolutionAction,
)
from twinscribe.models.components import (
    Component,
    ComponentLocation,
    ParameterDoc,
    ReturnDoc,
    ExceptionDoc,
    ComponentDocumentation,
)
from twinscribe.models.call_graph import (
    CallEdge,
    CallGraph,
    CallGraphDiff,
)
from twinscribe.models.documentation import (
    DocumenterMetadata,
    DocumentationOutput,
    CalleeRef,
    CallerRef,
    CallGraphSection,
)
from twinscribe.models.validation import (
    CompletenessCheck,
    CallGraphAccuracy,
    CorrectionApplied,
    ValidatorMetadata,
    ValidationResult,
)
from twinscribe.models.comparison import (
    Discrepancy,
    ConvergenceStatus,
    ComparisonSummary,
    ComparatorMetadata,
    ComparisonResult,
)
from twinscribe.models.convergence import (
    ConvergenceCriteria,
    ConvergenceHistoryEntry,
    ConvergenceReport,
)
from twinscribe.models.output import (
    ComponentFinalDoc,
    DocumentationPackage,
    RunMetrics,
    CostBreakdown,
)
from twinscribe.models.beads import (
    BeadsTicketPriority,
    BeadsTicketType,
    DiscrepancyTicket,
    RebuildTicket,
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
