"""
Documenter Agent Interface.

Defines the interface for documenter agents (A1, B1) that generate
comprehensive documentation with call graph linkages.

Reference: Spec section 3.1
"""

from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from twinscribe.agents.base import AgentConfig, BaseAgent
from twinscribe.models.base import ModelTier, StreamId
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import Component
from twinscribe.models.documentation import DocumentationOutput


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
    """

    component: Component = Field(
        ..., description="Component to document"
    )
    source_code: str = Field(
        ..., description="Source code of the component"
    )
    dependency_context: dict[str, DocumentationOutput] = Field(
        default_factory=dict,
        description="Documentation of dependencies (component_id -> doc)",
    )
    static_analysis_hints: Optional[CallGraph] = Field(
        default=None,
        description="Static analysis hints for guidance",
    )
    iteration: int = Field(
        default=1,
        ge=1,
        description="Current iteration number",
    )
    previous_output: Optional[DocumentationOutput] = Field(
        default=None,
        description="Previous documentation if re-documenting",
    )
    corrections: list[dict] = Field(
        default_factory=list,
        description="Corrections from validation to apply",
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
    SYSTEM_PROMPT = """You are a code documentation agent. Your task is to generate comprehensive
documentation for code components including accurate call graph linkages.

CRITICAL REQUIREMENTS:
1. Document ALL parameters, return values, and exceptions
2. Identify ALL function/method calls made BY this component (callees)
3. When available, identify callers OF this component
4. Be precise about call site line numbers
5. Distinguish between direct calls, conditional calls, and calls in loops

You have access to:
- The source code of the focal component
- Documentation of dependencies (already processed)
- Optional static analysis hints

Output in the specified JSON schema. Do not hallucinate call relationships
that don't exist in the code."""

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
            f"## Component to Document",
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
            lines.extend([
                "## Existing Docstring",
                "```",
                input_data.component.existing_docstring,
                "```",
                "",
            ])

        if input_data.dependency_context:
            lines.append("## Dependency Context (Already Documented)")
            for dep_id, dep_doc in input_data.dependency_context.items():
                lines.extend([
                    f"### {dep_id}",
                    f"Summary: {dep_doc.documentation.summary}",
                    "",
                ])

        if input_data.static_analysis_hints:
            lines.extend([
                "## Static Analysis Hints",
                f"Known callees: {len(input_data.static_analysis_hints.get_callees(input_data.component.component_id))}",
                f"Known callers: {len(input_data.static_analysis_hints.get_callers(input_data.component.component_id))}",
                "",
            ])

        if input_data.corrections:
            lines.extend([
                "## Corrections to Apply",
                "The following corrections were identified by validation:",
                "",
            ])
            for correction in input_data.corrections:
                lines.append(f"- {correction}")

        lines.extend([
            "",
            "Generate comprehensive documentation in the specified JSON format.",
        ])

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
