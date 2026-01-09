"""
Documenter Agent Interface.

Defines the interface for documenter agents (A1, B1) that generate
comprehensive documentation with call graph linkages.

Reference: Spec section 3.1
"""

from abc import abstractmethod

from pydantic import BaseModel, Field

from twinscribe.agents.base import AgentConfig, BaseAgent
from twinscribe.models.base import ModelTier, StreamId
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import Component
from twinscribe.models.documentation import DocumentationOutput
from twinscribe.models.feedback import CallGraphFeedback


class DocumenterInput(BaseModel):
    """Input to a documenter agent.

    Attributes:
        component: The code component to document
        source_code: Source code of the component
        dependency_context: Documentation of already-processed dependencies
        static_analysis_hints: Optional hints from static analysis
        iteration: Current iteration number (for re-documentation)
        previous_output: Previous documentation if re-documenting
        corrections: Corrections to apply from validation
        call_graph_feedback: Optional feedback about call graph discrepancies
            from the other stream, to help convergence
    """

    component: Component = Field(..., description="Component to document")
    source_code: str = Field(..., description="Source code of the component")
    dependency_context: dict[str, DocumentationOutput] = Field(
        default_factory=dict,
        description="Documentation of dependencies (component_id -> doc)",
    )
    static_analysis_hints: CallGraph | None = Field(
        default=None,
        description="Static analysis hints for guidance",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current iteration number",
    )
    previous_output: DocumentationOutput | None = Field(
        default=None,
        description="Previous documentation if re-documenting",
    )
    corrections: list[dict] = Field(
        default_factory=list,
        description="Corrections from validation to apply",
    )
    call_graph_feedback: list[CallGraphFeedback] = Field(
        default_factory=list,
        description="Feedback about call graph discrepancies from other stream",
    )


class DocumenterConfig(AgentConfig):
    """Configuration specific to documenter agents.

    Extends AgentConfig with documenter-specific settings.

    Attributes:
        include_examples: Whether to generate usage examples
        max_example_count: Maximum number of examples to generate
        include_see_also: Whether to include related references
        confidence_threshold: Minimum confidence to accept output
    """

    include_examples: bool = Field(
        default=True,
        description="Generate usage examples",
    )
    max_example_count: int = Field(
        default=3,
        ge=0,
        description="Maximum examples to generate",
    )
    include_see_also: bool = Field(
        default=True,
        description="Include related references",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable confidence",
    )


# Default configurations for documenter agents
STREAM_A_DOCUMENTER_CONFIG = DocumenterConfig(
    agent_id="A1",
    stream_id=StreamId.STREAM_A,
    model_tier=ModelTier.GENERATION,
    provider="anthropic",
    model_name="claude-sonnet-4-5-20250929",
    cost_per_million_input=3.0,
    cost_per_million_output=15.0,
    max_tokens=4096,
    temperature=0.0,
)

STREAM_B_DOCUMENTER_CONFIG = DocumenterConfig(
    agent_id="B1",
    stream_id=StreamId.STREAM_B,
    model_tier=ModelTier.GENERATION,
    provider="openai",
    model_name="gpt-4o",
    cost_per_million_input=2.5,
    cost_per_million_output=10.0,
    max_tokens=4096,
    temperature=0.0,
)


class DocumenterAgent(BaseAgent[DocumenterInput, DocumentationOutput]):
    """Abstract base class for documenter agents.

    Documenter agents generate comprehensive documentation for code
    components, including call graph linkages. Two independent agents
    run in parallel streams (A and B) using different models.

    The agent receives:
    - Source code of the focal component
    - Documentation of already-processed dependencies
    - Optional static analysis hints for guidance

    The agent produces:
    - Structured documentation (summary, description, parameters, etc.)
    - Call graph section (callers and callees)
    - Confidence score

    Reference: Spec section 3.1
    """

    # System prompt template for documenter agents
    SYSTEM_PROMPT = """You are a code documentation agent. Your task is to generate comprehensive,
high-quality documentation for code components including accurate call graph linkages.

DOCUMENTATION QUALITY REQUIREMENTS:

1. SUMMARY (1-2 sentences):
   - Capture the core purpose and intent of the component
   - Explain WHAT the component does at a high level
   - Should be meaningful and specific, not just restating the function name
   - Example: "Validates user authentication tokens against the session store and returns the associated user profile if valid."

2. DESCRIPTION (detailed, multi-paragraph):
   - WHAT: Explain the functionality in detail - what operations are performed, what data is processed
   - WHY: Explain the purpose and intent - why does this code exist? What problem does it solve?
   - HOW: Explain the algorithm, logic flow, or implementation approach - how does it achieve its goal?
   - WHEN: Explain the usage context - when should this be called? What are the prerequisites?
   - The description should help someone understand the code WITHOUT reading the source
   - Minimum 2-3 sentences, more for complex components
   - Include important edge cases, constraints, or side effects

3. PARAMETERS, RETURNS, AND EXCEPTIONS:
   - Document ALL parameters with type, purpose, and any constraints
   - Document return values with type and what the returned value represents
   - Document ALL exceptions that can be raised and under what conditions

4. CALL GRAPH (CRITICAL - DO NOT SKIP):
   CALLEES (what THIS component calls):
   - Scan EVERY line of source code for function/method calls
   - Include: function calls like foo(), method calls like obj.method(), constructor calls like ClassName()
   - Include: super().__init__(), self.other_method(), imported_module.function()
   - Include: attribute access that invokes properties: obj.property_name
   - Include: built-in calls: len(), str(), list(), dict(), print(), etc.
   - For abstract methods with no body: callees should be empty (this is correct)

   CALLERS (what calls THIS component):
   - Check the Dependency Context section - if component X is listed there and its summary mentions calling this component, X is a caller
   - Check if this is a method that would be called by other methods in the same class
   - Check if this is inherited/overridden - parent class methods are callers
   - For public methods: assume they have callers even if not explicitly known

   CALL TYPES:
   - "direct": Normal function/method call
   - "conditional": Call inside if/else/try/except
   - "loop": Call inside for/while loop
   - "callback": Passed as argument to another function

You have access to:
- The source code of the focal component
- Documentation of dependencies (already processed) - USE THIS TO IDENTIFY CALLERS
- Optional static analysis hints

Do not hallucinate call relationships that don't exist in the code.
Ensure descriptions are thorough and informative.

OUTPUT FORMAT - You MUST respond with valid JSON matching this exact structure:
{
  "documentation": {
    "summary": "One-line description (max 200 chars)",
    "description": "Detailed multi-paragraph explanation",
    "parameters": [
      {"name": "param_name", "type": "str", "description": "What it does", "required": true}
    ],
    "returns": {"type": "ReturnType", "description": "What is returned"},
    "raises": [{"type": "ExceptionType", "condition": "When raised"}],
    "examples": ["example code snippet"]
  },
  "call_graph": {
    "callers": [
      {"component_id": "module.Class.method_that_calls_this", "call_site_line": 42, "call_type": "direct"}
    ],
    "callees": [
      {"component_id": "module.function_this_calls", "call_site_line": 15, "call_type": "direct"}
    ]
  },
  "confidence": 0.85
}

CRITICAL REQUIREMENTS FOR call_graph:
1. NEVER return empty callers AND empty callees unless this is truly an isolated component
2. For abstract methods: callees=[] is correct, but look for callers in dependency context
3. For concrete methods: scan EVERY line for any function/method/constructor calls
4. Include built-in calls (len, str, list, dict, print, isinstance, etc.)
5. Include self.method() calls to other methods in the same class
6. If static_analysis_hints show callees, verify them against source code
7. call_site_line should be the actual line number from source code, or null if unknown"""

    def __init__(self, config: DocumenterConfig) -> None:
        """Initialize the documenter agent.

        Args:
            config: Documenter configuration
        """
        super().__init__(config)
        self._documenter_config = config

    @property
    def documenter_config(self) -> DocumenterConfig:
        """Get documenter-specific configuration."""
        return self._documenter_config

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the documenter agent.

        Sets up the LLM client and any required resources.
        """
        pass

    @abstractmethod
    async def process(self, input_data: DocumenterInput) -> DocumentationOutput:
        """Generate documentation for a component.

        Args:
            input_data: Input containing component, source, and context

        Returns:
            Structured documentation with call graph

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the documenter agent."""
        pass

    def _build_user_prompt(self, input_data: DocumenterInput) -> str:
        """Build the user prompt for the LLM.

        Args:
            input_data: Documenter input

        Returns:
            Formatted user prompt string
        """
        lines = [
            "## Component to Document",
            f"**ID:** {input_data.component.component_id}",
            f"**Type:** {input_data.component.type.value}",
            f"**Location:** {input_data.component.location.to_reference()}",
            "",
            "## Source Code",
            "```python",
            input_data.source_code,
            "```",
            "",
        ]

        if input_data.component.existing_docstring:
            lines.extend(
                [
                    "## Existing Docstring",
                    "```",
                    input_data.component.existing_docstring,
                    "```",
                    "",
                ]
            )

        if input_data.dependency_context:
            lines.append("## Dependency Context (Already Documented)")
            for dep_id, dep_doc in input_data.dependency_context.items():
                lines.extend(
                    [
                        f"### {dep_id}",
                        f"Summary: {dep_doc.documentation.summary}",
                        "",
                    ]
                )

        if input_data.static_analysis_hints:
            callees = input_data.static_analysis_hints.get_callees(
                input_data.component.component_id
            )
            callers = input_data.static_analysis_hints.get_callers(
                input_data.component.component_id
            )
            lines.extend(
                [
                    "## Static Analysis Hints (verify against source code)",
                    f"Known callees ({len(callees)}): {', '.join(callees[:20]) if callees else 'none detected'}",
                    f"Known callers ({len(callers)}): {', '.join(callers[:20]) if callers else 'none detected'}",
                    "",
                ]
            )
            if len(callees) > 20:
                lines.append(f"  ... and {len(callees) - 20} more callees")
            if len(callers) > 20:
                lines.append(f"  ... and {len(callers) - 20} more callers")

        if input_data.corrections:
            lines.extend(
                [
                    "## Corrections to Apply",
                    "The following corrections were identified by validation:",
                    "",
                ]
            )
            for correction in input_data.corrections:
                lines.append(f"- {correction}")

        # Add call graph feedback section if present
        if input_data.call_graph_feedback:
            lines.extend(
                [
                    "",
                    "## Call Graph Feedback from Other Stream",
                    "The following discrepancies were found between your call graph "
                    "and the other documentation stream. Please verify and update:",
                    "",
                ]
            )
            for feedback in input_data.call_graph_feedback:
                if feedback.has_feedback():
                    lines.append(f"### Component: {feedback.component_id}")
                    prompt_section = feedback.to_prompt_section()
                    if prompt_section:
                        lines.append(prompt_section)
                    lines.append("")

        lines.extend(
            [
                "",
                "Generate comprehensive documentation in the specified JSON format.",
            ]
        )

        return "\n".join(lines)

    def _get_response_schema(self) -> dict:
        """Get the JSON schema for structured output.

        Returns:
            JSON schema dict for DocumentationOutput
        """
        return {
            "type": "object",
            "properties": {
                "component_id": {"type": "string"},
                "documentation": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": ["string", "null"]},
                                    "description": {"type": "string"},
                                    "default": {"type": ["string", "null"]},
                                    "required": {"type": "boolean"},
                                },
                                "required": ["name"],
                            },
                        },
                        "returns": {
                            "type": ["object", "null"],
                            "properties": {
                                "type": {"type": ["string", "null"]},
                                "description": {"type": "string"},
                            },
                        },
                        "raises": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "condition": {"type": "string"},
                                },
                                "required": ["type"],
                            },
                        },
                        "examples": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["summary", "description"],
                },
                "call_graph": {
                    "type": "object",
                    "properties": {
                        "callers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component_id": {"type": "string"},
                                    "call_site_line": {"type": ["integer", "null"]},
                                    "call_type": {"type": "string"},
                                },
                                "required": ["component_id"],
                            },
                        },
                        "callees": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component_id": {"type": "string"},
                                    "call_site_line": {"type": ["integer", "null"]},
                                    "call_type": {"type": "string"},
                                },
                                "required": ["component_id"],
                            },
                        },
                    },
                },
            },
            "required": ["component_id", "documentation"],
        }
