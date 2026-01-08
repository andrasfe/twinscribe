"""
Base enumerations and constants used throughout the data models.

These enums provide type-safe values for common categorical fields
and ensure consistency across the system.
"""

from enum import Enum


class ComponentType(str, Enum):
    """Type of code component being documented."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    PROPERTY = "property"
    STATICMETHOD = "staticmethod"
    CLASSMETHOD = "classmethod"


class CallType(str, Enum):
    """Type of call relationship between components.

    Distinguishes between different invocation patterns
    which affects documentation and dependency analysis.
    """

    DIRECT = "direct"  # Simple function/method call
    CONDITIONAL = "conditional"  # Call within if/else branch
    LOOP = "loop"  # Call within for/while loop
    EXCEPTION = "exception"  # Call within try/except block
    CALLBACK = "callback"  # Passed as callback/closure
    DYNAMIC = "dynamic"  # Dynamic dispatch (getattr, etc.)


class DiscrepancyType(str, Enum):
    """Type of discrepancy between documentation streams.

    Categorization helps determine appropriate resolution strategy.
    """

    # Call graph related - can be resolved by static analysis
    CALL_GRAPH_EDGE = "call_graph_edge"  # Missing/extra call edge
    CALL_SITE_LINE = "call_site_line"  # Different line number
    CALL_TYPE_MISMATCH = "call_type_mismatch"  # Different call type

    # Documentation content - may require human review
    DOCUMENTATION_CONTENT = "documentation_content"  # Summary/description differs
    PARAMETER_DESCRIPTION = "parameter_description"  # Parameter doc differs
    RETURN_DESCRIPTION = "return_description"  # Return doc differs
    EXCEPTION_DOCUMENTATION = "exception_documentation"  # Exception doc differs

    # Structural differences
    MISSING_PARAMETER = "missing_parameter"  # One stream missing param doc
    MISSING_EXCEPTION = "missing_exception"  # One stream missing exception doc
    TYPE_ANNOTATION = "type_annotation"  # Type annotation differs


class ValidationStatus(str, Enum):
    """Result status from validation agent."""

    PASS = "pass"  # All checks passed
    FAIL = "fail"  # Critical validation failures
    WARNING = "warning"  # Non-blocking issues found


class ResolutionAction(str, Enum):
    """How a discrepancy was or should be resolved."""

    ACCEPT_STREAM_A = "accept_stream_a"
    ACCEPT_STREAM_B = "accept_stream_b"
    ACCEPT_GROUND_TRUTH = "accept_ground_truth"
    ACCEPT_CONSENSUS = "accept_consensus"  # Both streams agree
    MERGE_BOTH = "merge_both"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    DEFERRED = "deferred"


class ResolutionSource(str, Enum):
    """Source of the resolution for a discrepancy.

    Tracks whether the resolution came from consensus (both streams agree),
    ground truth hints, or human review. This is used for auditing and
    to understand how discrepancies were resolved.
    """

    CONSENSUS = "consensus"  # Both streams agreed (A == B)
    GROUND_TRUTH_HINT = "ground_truth_hint"  # Resolved using static analysis as hint
    HUMAN_REVIEW = "human_review"  # Resolved by human via Beads
    AUTO_RESOLVED = "auto_resolved"  # System auto-resolved (e.g., one stream missing)
    UNRESOLVED = "unresolved"  # Not yet resolved


class ModelTier(str, Enum):
    """Tier of model in the cost hierarchy."""

    GENERATION = "generation"  # High-quality, mid-cost
    VALIDATION = "validation"  # Fast, cheap
    ARBITRATION = "arbitration"  # Premium, expensive


class StreamId(str, Enum):
    """Identifier for documentation streams."""

    STREAM_A = "A"
    STREAM_B = "B"
