"""
Validation result models from validator agents.

These models define the output schema produced by validator agents
(A2 and B2) as specified in section 3.2 of the specification.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from twinscribe.models.base import StreamId, ValidationStatus


class CompletenessCheck(BaseModel):
    """Result of documentation completeness validation.

    Checks whether all code elements are properly documented.

    Attributes:
        score: Completeness score from 0.0 to 1.0
        missing_elements: Documentation gaps found
        extra_elements: Documented items not in code
    """

    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Completeness score",
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Missing documentation elements",
        examples=[["exception: RuntimeError not documented"]],
    )
    extra_elements: list[str] = Field(
        default_factory=list,
        description="Extra elements not in code",
        examples=[["parameter 'debug' not in signature"]],
    )

    @computed_field
    @property
    def is_complete(self) -> bool:
        """True if no missing elements."""
        return len(self.missing_elements) == 0


class CallGraphAccuracy(BaseModel):
    """Result of call graph validation against static analysis.

    Compares documented call relationships against ground truth
    from static analysis tools.

    Attributes:
        score: Accuracy score from 0.0 to 1.0
        verified_callees: Callees confirmed by static analysis
        missing_callees: Callees in static analysis but not documented
        false_callees: Documented callees not in static analysis
        verified_callers: Callers confirmed by static analysis
        missing_callers: Callers in static analysis but not documented
        false_callers: Documented callers not in static analysis
    """

    score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall accuracy score",
    )
    verified_callees: list[str] = Field(
        default_factory=list,
        description="Confirmed callee component IDs",
    )
    missing_callees: list[str] = Field(
        default_factory=list,
        description="Undocumented callees",
    )
    false_callees: list[str] = Field(
        default_factory=list,
        description="Incorrectly documented callees",
    )
    verified_callers: list[str] = Field(
        default_factory=list,
        description="Confirmed caller component IDs",
    )
    missing_callers: list[str] = Field(
        default_factory=list,
        description="Undocumented callers",
    )
    false_callers: list[str] = Field(
        default_factory=list,
        description="Incorrectly documented callers",
    )

    @computed_field
    @property
    def has_errors(self) -> bool:
        """True if any false positives or significant missing edges."""
        return (
            len(self.false_callees) > 0
            or len(self.false_callers) > 0
            or len(self.missing_callees) > 0
        )

    @computed_field
    @property
    def callee_precision(self) -> float:
        """Precision of callee documentation."""
        total = len(self.verified_callees) + len(self.false_callees)
        return len(self.verified_callees) / total if total > 0 else 1.0

    @computed_field
    @property
    def callee_recall(self) -> float:
        """Recall of callee documentation."""
        total = len(self.verified_callees) + len(self.missing_callees)
        return len(self.verified_callees) / total if total > 0 else 1.0


class CorrectionApplied(BaseModel):
    """Record of a correction applied by the validator.

    Validators can apply corrections when documentation conflicts
    with static analysis ground truth.

    Attributes:
        field: Field path that was corrected
        action: What was done (added, removed, modified)
        original_value: Value before correction
        corrected_value: Value after correction
        reason: Why correction was made
    """

    field: str = Field(
        ...,
        description="Field path corrected",
        examples=["call_graph.callees", "documentation.parameters"],
    )
    action: str = Field(
        ...,
        description="Action taken",
        examples=["added", "removed", "modified"],
    )
    original_value: Optional[str] = Field(
        default=None, description="Value before correction"
    )
    corrected_value: Optional[str] = Field(
        default=None, description="Value after correction"
    )
    reason: str = Field(
        default="",
        description="Explanation for correction",
        examples=["Not found in static analysis"],
    )


class ValidatorMetadata(BaseModel):
    """Metadata about the validation process.

    Attributes:
        agent_id: Identifier of the validator agent (A2, B2)
        stream_id: Which stream this agent belongs to
        model: Model name used for validation
        static_analyzer: Static analysis tool used for ground truth
        timestamp: When validation was performed
        token_count: Tokens consumed for validation
    """

    agent_id: str = Field(
        ...,
        description="Agent identifier",
        examples=["A2", "B2"],
    )
    stream_id: StreamId = Field(
        ..., description="Stream this agent belongs to"
    )
    model: str = Field(
        ...,
        description="Model name used",
        examples=["claude-haiku-4-5-20251001", "gpt-4o-mini"],
    )
    static_analyzer: str = Field(
        default="pycg",
        description="Static analysis tool used",
        examples=["pycg", "pyan3"],
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When validation occurred",
    )
    token_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Tokens consumed",
    )


class ValidationResult(BaseModel):
    """Complete output from a validator agent for one component.

    This is the main output schema for validator agents (A2, B2)
    as defined in spec section 3.2.

    Attributes:
        component_id: ID of the validated component
        validation_result: Overall pass/fail/warning status
        completeness: Documentation completeness check results
        call_graph_accuracy: Call graph validation results
        corrections_applied: List of corrections made
        metadata: Validator agent information
    """

    component_id: str = Field(
        ...,
        min_length=1,
        description="Validated component ID",
    )
    validation_result: ValidationStatus = Field(
        ..., description="Overall validation status"
    )
    completeness: CompletenessCheck = Field(
        default_factory=CompletenessCheck,
        description="Completeness check results",
    )
    call_graph_accuracy: CallGraphAccuracy = Field(
        default_factory=CallGraphAccuracy,
        description="Call graph accuracy results",
    )
    corrections_applied: list[CorrectionApplied] = Field(
        default_factory=list,
        description="Corrections made by validator",
    )
    metadata: ValidatorMetadata = Field(
        ..., description="Validator metadata"
    )

    @computed_field
    @property
    def is_valid(self) -> bool:
        """True if validation passed without failures."""
        return self.validation_result != ValidationStatus.FAIL

    @computed_field
    @property
    def requires_iteration(self) -> bool:
        """True if issues were found that might need re-documentation."""
        return (
            self.validation_result == ValidationStatus.FAIL
            or len(self.corrections_applied) > 0
        )

    @computed_field
    @property
    def total_corrections(self) -> int:
        """Number of corrections applied."""
        return len(self.corrections_applied)
