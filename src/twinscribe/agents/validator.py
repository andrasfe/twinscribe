"""
Validator Agent Interface.

Defines the interface for validator agents (A2, B2) that verify
documentation completeness and call graph accuracy against source code.

Reference: Spec section 3.2
"""

from abc import abstractmethod

from pydantic import BaseModel, Field

from twinscribe.agents.base import AgentConfig, BaseAgent
from twinscribe.models.base import ModelTier, StreamId
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.documentation import DocumentationOutput
from twinscribe.models.validation import ValidationResult


class ValidatorInput(BaseModel):
    """Input to a validator agent.

    Attributes:
        documentation: Documentation output to validate
        source_code: Original source code of the component
        ground_truth_call_graph: Optional static analysis call graph (hints, not authoritative)
        component_ast: Optional AST representation for detailed checks
    """

    documentation: DocumentationOutput = Field(..., description="Documentation to validate")
    source_code: str = Field(..., description="Original source code")
    ground_truth_call_graph: CallGraph | None = Field(
        default=None,
        description="Static analysis call graph (optional hints, not authoritative)",
    )
    component_ast: dict | None = Field(
        default=None,
        description="AST representation for detailed checks",
    )


class ValidatorConfig(AgentConfig):
    """Configuration specific to validator agents.

    Extends AgentConfig with validator-specific settings.

    Attributes:
        auto_correct: Whether to apply automatic corrections
        false_positive_threshold: Max false positive rate before fail
        missing_callee_threshold: Max missing callee rate before fail
        missing_caller_threshold: Max missing caller rate before warning
        require_all_parameters: Fail if any parameter undocumented
        require_return_doc: Fail if return value undocumented
        require_exception_doc: Fail if raisable exceptions undocumented
    """

    auto_correct: bool = Field(
        default=True,
        description="Apply automatic corrections",
    )
    false_positive_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Max false positive rate (fail threshold)",
    )
    missing_callee_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Max missing callee rate (fail threshold)",
    )
    missing_caller_threshold: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Max missing caller rate (warning threshold)",
    )
    require_all_parameters: bool = Field(
        default=True,
        description="Require all parameters documented",
    )
    require_return_doc: bool = Field(
        default=True,
        description="Require return value documented",
    )
    require_exception_doc: bool = Field(
        default=True,
        description="Require exceptions documented",
    )


# Default configurations for validator agents
STREAM_A_VALIDATOR_CONFIG = ValidatorConfig(
    agent_id="A2",
    stream_id=StreamId.STREAM_A,
    model_tier=ModelTier.VALIDATION,
    provider="anthropic",
    model_name="claude-haiku-4-5-20251001",
    cost_per_million_input=0.25,
    cost_per_million_output=1.25,
    max_tokens=2048,
    temperature=0.0,
)

STREAM_B_VALIDATOR_CONFIG = ValidatorConfig(
    agent_id="B2",
    stream_id=StreamId.STREAM_B,
    model_tier=ModelTier.VALIDATION,
    provider="openai",
    model_name="gpt-4o-mini",
    cost_per_million_input=0.15,
    cost_per_million_output=0.60,
    max_tokens=2048,
    temperature=0.0,
)


class ValidatorAgent(BaseAgent[ValidatorInput, ValidationResult]):
    """Abstract base class for validator agents.

    Validator agents verify documentation completeness and call graph
    accuracy against source code. Static analysis is used as optional
    hints, but the final source of truth is dual-stream consensus.

    The agent receives:
    - Documentation output from the documenter
    - Original source code
    - Optional static analysis call graph (hints, not authoritative)

    The agent produces:
    - Validation status (pass/fail/warning)
    - Completeness check results
    - Call graph accuracy results
    - List of corrections applied

    Note: Static analysis provides helpful hints but does NOT auto-correct.
    The dual-stream consensus (A == B agreement) is the authoritative mechanism.

    Reference: Spec section 3.2
    """

    # System prompt template for validator agents
    SYSTEM_PROMPT = """You are a documentation validation agent. Your task is to verify that
documentation is complete, high-quality, and that call graph linkages are reasonable.

You have access to:
1. The documentation to validate
2. The original source code
3. Static analysis call graph (optional hints for reference, NOT authoritative)

VALIDATION RULES:

1. DESCRIPTION QUALITY (Critical - check this carefully):
   - Summary must be non-empty and meaningful (not just restating the function name)
   - Description must be at least 2-3 sentences explaining functionality
   - Description must explain PURPOSE/INTENT, not just list parameters
   - Description should cover WHAT, WHY, HOW, and WHEN aspects
   - Description should help understand the code without reading source

   Flag these description quality issues:
   - "summary_empty": Summary is missing or empty
   - "summary_too_brief": Summary is just one or two words
   - "summary_restates_name": Summary just restates the function/class name
   - "description_empty": Description is missing or empty
   - "description_too_brief": Description is less than 2 sentences
   - "missing_purpose": Description doesn't explain WHY the code exists
   - "missing_functionality": Description doesn't explain WHAT the code does
   - "missing_usage_context": Description doesn't explain WHEN to use it
   - "description_restates_params": Description only lists parameters without explaining logic

2. COMPLETENESS:
   - All parameters in code must be documented
   - All return paths must be documented
   - All exceptions that can be raised must be documented

3. CALL GRAPH ACCURACY:
   - Validate that the documented call graph is reasonable based on the source code
   - If static analysis hints are available, use them as guidance but NOT as authority
   - The dual-stream consensus (agreement between Stream A and B) will be the true authority
   - Flag potential discrepancies but do not auto-correct based solely on static analysis

IMPORTANT: Static analysis provides helpful hints but should not override your judgment
based on the source code. The final source of truth is dual-stream consensus.

Output validation results in the specified JSON schema. Include description_quality
assessment with a score and list of specific issues found."""

    def __init__(self, config: ValidatorConfig) -> None:
        """Initialize the validator agent.

        Args:
            config: Validator configuration
        """
        super().__init__(config)
        self._validator_config = config

    @property
    def validator_config(self) -> ValidatorConfig:
        """Get validator-specific configuration."""
        return self._validator_config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the validator agent.

        Sets up the LLM client and any required resources.
        """
        pass

    @abstractmethod
    async def process(self, input_data: ValidatorInput) -> ValidationResult:
        """Validate documentation for a component.

        Args:
            input_data: Input containing documentation and ground truth

        Returns:
            Validation result with corrections

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the validator agent."""
        pass

    def _build_user_prompt(self, input_data: ValidatorInput) -> str:
        """Build the user prompt for validation.

        Args:
            input_data: Validator input

        Returns:
            Formatted user prompt string
        """
        doc = input_data.documentation
        gt = input_data.ground_truth_call_graph

        # Get ground truth edges for this component (if available)
        gt_callees = gt.get_callees(doc.component_id) if gt else []
        gt_callers = gt.get_callers(doc.component_id) if gt else []

        lines = [
            "## Documentation to Validate",
            f"**Component ID:** {doc.component_id}",
            "",
            "### Summary",
            doc.documentation.summary,
            "",
            "### Description",
            doc.documentation.description,
            "",
            "### Parameters",
        ]

        for param in doc.documentation.parameters:
            lines.append(f"- {param.name}: {param.type} - {param.description}")
        if not doc.documentation.parameters:
            lines.append("_None documented_")

        lines.extend(
            [
                "",
                "### Returns",
            ]
        )
        if doc.documentation.returns:
            lines.append(
                f"- {doc.documentation.returns.type}: {doc.documentation.returns.description}"
            )
        else:
            lines.append("_None documented_")

        lines.extend(
            [
                "",
                "### Exceptions",
            ]
        )
        for exc in doc.documentation.raises:
            lines.append(f"- {exc.type}: {exc.condition}")
        if not doc.documentation.raises:
            lines.append("_None documented_")

        lines.extend(
            [
                "",
                "### Documented Call Graph",
                "**Callees (this component calls):**",
            ]
        )
        for callee in doc.call_graph.callees:
            lines.append(
                f"- {callee.component_id} (line {callee.call_site_line}, {callee.call_type.value})"
            )
        if not doc.call_graph.callees:
            lines.append("_None_")

        lines.extend(
            [
                "",
                "**Callers (call this component):**",
            ]
        )
        for caller in doc.call_graph.callers:
            lines.append(f"- {caller.component_id} (line {caller.call_site_line})")
        if not doc.call_graph.callers:
            lines.append("_None_")

        lines.extend(
            [
                "",
                "## Source Code",
                "```python",
                input_data.source_code,
                "```",
            ]
        )

        # Only include static analysis hints if available
        if gt is not None:
            lines.extend(
                [
                    "",
                    "## Static Analysis Hints (for reference only, NOT authoritative)",
                    "**Callees:**",
                ]
            )
            for edge in gt_callees:
                lines.append(f"- {edge.callee} (line {edge.call_site_line})")
            if not gt_callees:
                lines.append("_None_")

            lines.extend(
                [
                    "",
                    "**Callers:**",
                ]
            )
            for edge in gt_callers:
                lines.append(f"- {edge.caller} (line {edge.call_site_line})")
            if not gt_callers:
                lines.append("_None_")

        lines.extend(
            [
                "",
                "## Validation Tasks",
                "",
                "### Description Quality (Critical)",
                "1. Check summary is non-empty and meaningful (not just restating function name)",
                "2. Check description is at least 2-3 sentences",
                "3. Check description explains PURPOSE (why the code exists)",
                "4. Check description explains FUNCTIONALITY (what it does)",
                "5. Check description explains USAGE CONTEXT (when to use it)",
                "6. Flag any quality issues found in description_quality.issues",
                "",
                "### Completeness",
                "7. Check all parameters are documented",
                "8. Check return value is documented (if function returns something)",
                "9. Check all exceptions are documented",
                "",
                "### Call Graph Accuracy",
                "10. Validate documented call graph is reasonable based on source code",
                "11. Use static analysis hints (if available) as guidance, not authority",
                "12. Flag potential issues but do NOT auto-correct based solely on hints",
                "",
                "IMPORTANT: Static analysis hints are for reference only.",
                "The final source of truth is dual-stream consensus (Stream A == Stream B).",
                "",
                "Output validation results in the specified JSON format.",
            ]
        )

        return "\n".join(lines)

    def _get_response_schema(self) -> dict:
        """Get the JSON schema for validation output.

        Returns:
            JSON schema dict for ValidationResult
        """
        return {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "validation_result": {
                    "type": "string",
                    "enum": ["pass", "fail", "warning"],
                },
                "completeness": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "missing_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "extra_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "call_graph_accuracy": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "verified_callees": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "missing_callees": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "false_callees": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "verified_callers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "missing_callers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "false_callers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "description_quality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "issues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Quality issues like: summary_empty, summary_too_brief, "
                            "summary_restates_name, description_empty, description_too_brief, "
                            "missing_purpose, missing_functionality, missing_usage_context, "
                            "description_restates_params",
                        },
                    },
                },
                "corrections_applied": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "action": {"type": "string"},
                            "original_value": {"type": ["string", "null"]},
                            "corrected_value": {"type": ["string", "null"]},
                            "reason": {"type": "string"},
                        },
                        "required": ["field", "action", "reason"],
                    },
                },
            },
            "required": ["component_id", "validation_result"],
        }
