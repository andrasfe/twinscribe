# Agent Architecture Design Document

## Version 1.0 | January 2026

---

## 1. Overview

This document specifies the architecture for the documenter agents (A1, B1) and validator agents (A2, B2) in the Dual-Stream Code Documentation System. The design follows a tiered model approach where generation-tier models produce documentation and validation-tier models verify accuracy against static analysis ground truth.

### Design Goals

1. **Separation of Concerns**: Clear boundaries between documentation generation and validation
2. **Stream Independence**: Each stream operates autonomously with its own model configuration
3. **Extensibility**: Easy to add new agent types or swap model providers
4. **Testability**: All components are unit-testable with dependency injection
5. **Type Safety**: Full type hints and Pydantic models for data validation

---

## 2. Class Hierarchy

```
                                    +------------------+
                                    |   BaseAgent      |
                                    |   <<abstract>>   |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                                                 |
          +---------+---------+                            +----------+----------+
          | DocumenterAgent   |                            | ValidatorAgent      |
          | <<abstract>>      |                            | <<abstract>>        |
          +--------+----------+                            +----------+----------+
                   |                                                  |
       +-----------+-----------+                          +-----------+-----------+
       |                       |                          |                       |
+------+------+         +------+------+            +------+------+         +------+------+
| ClaudeDoc   |         | OpenAIDoc   |            | ClaudeVal   |         | OpenAIVal   |
| Agent       |         | Agent       |            | Agent       |         | Agent       |
+-------------+         +-------------+            +-------------+         +-------------+
```

---

## 3. Core Interfaces and Abstract Classes

### 3.1 AgentInput Protocol

```python
# twinscribe/agents/protocols.py

from typing import Protocol, runtime_checkable, Any
from dataclasses import dataclass


@runtime_checkable
class AgentInput(Protocol):
    """Protocol defining the contract for agent input data."""

    def to_prompt_context(self) -> str:
        """Convert input to context string for LLM prompt."""
        ...

    def get_component_id(self) -> str:
        """Return the unique identifier for the component being processed."""
        ...


@runtime_checkable
class AgentOutput(Protocol):
    """Protocol defining the contract for agent output data."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize output to dictionary."""
        ...

    def get_component_id(self) -> str:
        """Return the component ID this output pertains to."""
        ...

    def get_confidence(self) -> float:
        """Return confidence score (0.0 to 1.0)."""
        ...
```

### 3.2 BaseAgent Abstract Class

```python
# twinscribe/agents/base.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from datetime import datetime
import asyncio

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from twinscribe.llm.client import LLMClient
from twinscribe.config.models import ModelConfig


InputT = TypeVar('InputT', bound=AgentInput)
OutputT = TypeVar('OutputT', bound=AgentOutput)


class AgentMetadata(BaseModel):
    """Metadata attached to every agent operation."""
    agent_id: str
    agent_type: str
    model: str
    provider: str
    timestamp: datetime
    processing_time_ms: int
    token_usage: dict[str, int]
    retry_count: int = 0


class BaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all agents in the documentation system.

    Responsibilities:
    - Manage LLM client lifecycle
    - Handle retries and error recovery
    - Track metadata and token usage
    - Provide consistent interface across agent types

    Type Parameters:
        InputT: The input type this agent accepts
        OutputT: The output type this agent produces
    """

    def __init__(
        self,
        agent_id: str,
        model_config: ModelConfig,
        llm_client: LLMClient,
        system_prompt: str,
        max_retries: int = 3,
    ):
        self.agent_id = agent_id
        self.model_config = model_config
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self._total_tokens_used = 0
        self._operations_count = 0

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the type identifier for this agent (e.g., 'documenter', 'validator')."""
        ...

    @abstractmethod
    async def _build_prompt(self, input_data: InputT) -> str:
        """
        Build the user prompt from input data.

        Args:
            input_data: The input to convert to a prompt

        Returns:
            The formatted prompt string
        """
        ...

    @abstractmethod
    def _parse_response(self, response: str, input_data: InputT) -> OutputT:
        """
        Parse the LLM response into the expected output type.

        Args:
            response: Raw LLM response string
            input_data: Original input (for context in parsing)

        Returns:
            Parsed and validated output

        Raises:
            OutputParseError: If response cannot be parsed
        """
        ...

    @abstractmethod
    def _validate_output(self, output: OutputT) -> list[str]:
        """
        Validate the parsed output for completeness and correctness.

        Args:
            output: The parsed output to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        ...

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def process(self, input_data: InputT) -> tuple[OutputT, AgentMetadata]:
        """
        Process input and produce output.

        This is the main entry point for agent processing. It handles:
        - Prompt construction
        - LLM invocation with retries
        - Response parsing
        - Output validation
        - Metadata collection

        Args:
            input_data: The input to process

        Returns:
            Tuple of (output, metadata)

        Raises:
            AgentProcessingError: If processing fails after retries
        """
        start_time = datetime.now()
        retry_count = 0

        try:
            prompt = await self._build_prompt(input_data)

            response = await self.llm_client.complete(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                model=self.model_config.model_name,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=4096,
            )

            output = self._parse_response(response.content, input_data)

            validation_errors = self._validate_output(output)
            if validation_errors:
                raise OutputValidationError(
                    f"Output validation failed: {validation_errors}"
                )

            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

            self._total_tokens_used += response.usage.total_tokens
            self._operations_count += 1

            metadata = AgentMetadata(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                model=self.model_config.model_name,
                provider=self.model_config.provider,
                timestamp=end_time,
                processing_time_ms=processing_time_ms,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                retry_count=retry_count,
            )

            return output, metadata

        except Exception as e:
            retry_count += 1
            if retry_count >= self.max_retries:
                raise AgentProcessingError(
                    f"Agent {self.agent_id} failed after {retry_count} retries: {e}"
                ) from e
            raise  # Re-raise for retry decorator

    def get_statistics(self) -> dict:
        """Return agent usage statistics."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_tokens_used": self._total_tokens_used,
            "operations_count": self._operations_count,
            "estimated_cost": self._calculate_cost(),
        }

    def _calculate_cost(self) -> float:
        """Calculate estimated cost based on token usage."""
        return (self._total_tokens_used / 1_000_000) * self.model_config.cost_per_million
```

---

## 4. Documenter Agent Architecture

### 4.1 Data Models

```python
# twinscribe/agents/documenter/models.py

from datetime import datetime
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class CallType(str, Enum):
    """Type of function/method call."""
    DIRECT = "direct"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    EXCEPTION_HANDLER = "exception_handler"
    CONTEXT_MANAGER = "context_manager"


class ParameterDoc(BaseModel):
    """Documentation for a function/method parameter."""
    name: str
    type: Optional[str] = None
    description: str
    default: Optional[str] = None
    is_required: bool = True


class ReturnDoc(BaseModel):
    """Documentation for a return value."""
    type: str
    description: str


class ExceptionDoc(BaseModel):
    """Documentation for an exception that can be raised."""
    type: str
    condition: str


class CallReference(BaseModel):
    """Reference to a called or calling component."""
    component_id: str
    call_site_line: int
    call_type: CallType = CallType.DIRECT


class DocumentationContent(BaseModel):
    """The actual documentation content."""
    summary: str = Field(..., min_length=10, max_length=200)
    description: str = Field(..., min_length=20)
    parameters: list[ParameterDoc] = Field(default_factory=list)
    returns: Optional[ReturnDoc] = None
    raises: list[ExceptionDoc] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    see_also: list[str] = Field(default_factory=list)


class CallGraph(BaseModel):
    """Call graph information for a component."""
    callers: list[CallReference] = Field(default_factory=list)
    callees: list[CallReference] = Field(default_factory=list)


class DocumenterInput(BaseModel):
    """Input to the documenter agent."""
    component_id: str
    source_code: str
    file_path: str
    line_start: int
    line_end: int
    component_type: str  # 'function', 'method', 'class'
    dependency_context: dict[str, DocumentationContent] = Field(default_factory=dict)
    static_analysis_hints: Optional[CallGraph] = None

    def to_prompt_context(self) -> str:
        """Convert to prompt context string."""
        context_parts = [
            f"Component ID: {self.component_id}",
            f"Component Type: {self.component_type}",
            f"File: {self.file_path}:{self.line_start}-{self.line_end}",
            "",
            "Source Code:",
            "```",
            self.source_code,
            "```",
        ]

        if self.dependency_context:
            context_parts.append("\nDependency Documentation:")
            for dep_id, doc in self.dependency_context.items():
                context_parts.append(f"\n{dep_id}:")
                context_parts.append(f"  Summary: {doc.summary}")

        if self.static_analysis_hints:
            context_parts.append("\nStatic Analysis Hints:")
            if self.static_analysis_hints.callees:
                context_parts.append("  Known callees: " +
                    ", ".join(c.component_id for c in self.static_analysis_hints.callees))

        return "\n".join(context_parts)

    def get_component_id(self) -> str:
        return self.component_id


class DocumenterOutput(BaseModel):
    """Output from the documenter agent."""
    component_id: str
    documentation: DocumentationContent
    call_graph: CallGraph
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_component_id(self) -> str:
        return self.component_id

    def get_confidence(self) -> float:
        return self.metadata.get("confidence", 0.0)
```

### 4.2 DocumenterAgent Implementation

```python
# twinscribe/agents/documenter/agent.py

import json
import re
from typing import Optional

from twinscribe.agents.base import BaseAgent
from twinscribe.agents.documenter.models import (
    DocumenterInput,
    DocumenterOutput,
    DocumentationContent,
    CallGraph,
)
from twinscribe.agents.prompts import DOCUMENTER_SYSTEM_PROMPT, DOCUMENTER_USER_TEMPLATE


class DocumenterAgent(BaseAgent[DocumenterInput, DocumenterOutput]):
    """
    Agent responsible for generating documentation with call graph linkages.

    Model Tier: Generation (Claude Sonnet 4.5 / GPT-4o)

    Responsibilities:
    - Generate comprehensive documentation for code components
    - Identify all function/method calls (callees)
    - Document parameters, returns, and exceptions
    - Produce structured JSON output conforming to schema

    The documenter operates in topological order, receiving documentation
    of dependencies as context to improve accuracy.
    """

    @property
    def agent_type(self) -> str:
        return "documenter"

    async def _build_prompt(self, input_data: DocumenterInput) -> str:
        """Build the documentation prompt."""
        return DOCUMENTER_USER_TEMPLATE.format(
            context=input_data.to_prompt_context(),
            component_type=input_data.component_type,
            component_id=input_data.component_id,
        )

    def _parse_response(
        self,
        response: str,
        input_data: DocumenterInput
    ) -> DocumenterOutput:
        """Parse LLM response into DocumenterOutput."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to parse entire response as JSON
            json_str = response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise OutputParseError(f"Failed to parse JSON response: {e}")

        # Validate and construct output
        try:
            documentation = DocumentationContent(**data.get("documentation", {}))
            call_graph = CallGraph(**data.get("call_graph", {}))

            return DocumenterOutput(
                component_id=input_data.component_id,
                documentation=documentation,
                call_graph=call_graph,
                metadata={
                    "confidence": data.get("metadata", {}).get("confidence", 0.85),
                    "processing_order": data.get("metadata", {}).get("processing_order"),
                },
            )
        except Exception as e:
            raise OutputParseError(f"Failed to construct output: {e}")

    def _validate_output(self, output: DocumenterOutput) -> list[str]:
        """Validate documentation completeness."""
        errors = []

        # Check summary
        if not output.documentation.summary:
            errors.append("Missing summary")

        # Check description
        if not output.documentation.description:
            errors.append("Missing description")

        # Validate call graph entries have required fields
        for callee in output.call_graph.callees:
            if not callee.component_id:
                errors.append("Callee missing component_id")
            if callee.call_site_line < 0:
                errors.append(f"Invalid call site line for {callee.component_id}")

        return errors


class ClaudeDocumenterAgent(DocumenterAgent):
    """Documenter agent using Claude Sonnet 4.5."""

    def __init__(self, agent_id: str, llm_client: LLMClient, **kwargs):
        model_config = ModelConfig(
            tier=ModelTier.GENERATION,
            provider="anthropic",
            model_name="claude-sonnet-4-5-20250929",
            cost_per_million=3.0,
        )
        super().__init__(
            agent_id=agent_id,
            model_config=model_config,
            llm_client=llm_client,
            system_prompt=DOCUMENTER_SYSTEM_PROMPT,
            **kwargs,
        )


class OpenAIDocumenterAgent(DocumenterAgent):
    """Documenter agent using GPT-4o."""

    def __init__(self, agent_id: str, llm_client: LLMClient, **kwargs):
        model_config = ModelConfig(
            tier=ModelTier.GENERATION,
            provider="openai",
            model_name="gpt-4o",
            cost_per_million=2.5,
        )
        super().__init__(
            agent_id=agent_id,
            model_config=model_config,
            llm_client=llm_client,
            system_prompt=DOCUMENTER_SYSTEM_PROMPT,
            **kwargs,
        )
```

---

## 5. Validator Agent Architecture

### 5.1 Data Models

```python
# twinscribe/agents/validator/models.py

from datetime import datetime
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


class ValidationResult(str, Enum):
    """Result of validation check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class CompletenessScore(BaseModel):
    """Score for documentation completeness."""
    score: float = Field(..., ge=0.0, le=1.0)
    missing_elements: list[str] = Field(default_factory=list)
    extra_elements: list[str] = Field(default_factory=list)


class CallGraphAccuracy(BaseModel):
    """Accuracy assessment for call graph."""
    score: float = Field(..., ge=0.0, le=1.0)
    verified_callees: list[str] = Field(default_factory=list)
    missing_callees: list[str] = Field(default_factory=list)
    false_callees: list[str] = Field(default_factory=list)
    verified_callers: list[str] = Field(default_factory=list)
    missing_callers: list[str] = Field(default_factory=list)
    false_callers: list[str] = Field(default_factory=list)


class CorrectionAction(str, Enum):
    """Type of correction applied."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class Correction(BaseModel):
    """A correction applied during validation."""
    field: str
    action: CorrectionAction
    value: str
    reason: str


class ValidatorInput(BaseModel):
    """Input to the validator agent."""
    component_id: str
    documentation_output: dict  # DocumenterOutput as dict
    source_code: str
    static_call_graph: dict  # Ground truth from static analysis

    def to_prompt_context(self) -> str:
        """Convert to prompt context string."""
        return f"""
Component ID: {self.component_id}

Documentation to Validate:
{json.dumps(self.documentation_output, indent=2)}

Source Code:
```
{self.source_code}
```

Static Analysis Call Graph (Ground Truth):
{json.dumps(self.static_call_graph, indent=2)}
"""

    def get_component_id(self) -> str:
        return self.component_id


class ValidatorOutput(BaseModel):
    """Output from the validator agent."""
    component_id: str
    validation_result: ValidationResult
    completeness: CompletenessScore
    call_graph_accuracy: CallGraphAccuracy
    corrections_applied: list[Correction] = Field(default_factory=list)
    corrected_documentation: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_component_id(self) -> str:
        return self.component_id

    def get_confidence(self) -> float:
        # Confidence based on validation scores
        return (self.completeness.score + self.call_graph_accuracy.score) / 2
```

### 5.2 ValidatorAgent Implementation

```python
# twinscribe/agents/validator/agent.py

import json
import re
from typing import Optional

from twinscribe.agents.base import BaseAgent
from twinscribe.agents.validator.models import (
    ValidatorInput,
    ValidatorOutput,
    ValidationResult,
    CompletenessScore,
    CallGraphAccuracy,
    Correction,
)
from twinscribe.agents.prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_TEMPLATE


class ValidatorAgent(BaseAgent[ValidatorInput, ValidatorOutput]):
    """
    Agent responsible for validating documentation against source code
    and static analysis ground truth.

    Model Tier: Validation (Claude Haiku 4.5 / GPT-4o-mini)

    Responsibilities:
    - Verify parameter completeness against source code
    - Validate return type documentation
    - Check call graph accuracy against static analysis
    - Apply corrections when discrepancies found
    - Trust static analysis as ground truth

    Validation Thresholds:
    - Parameter completeness: Any missing = fail
    - Return type: Must be documented if function returns value
    - Callees accuracy: >5% false positive = fail
    - Callers accuracy: >10% missing = fail
    """

    # Validation thresholds
    CALLEE_FALSE_POSITIVE_THRESHOLD = 0.05
    CALLER_MISSING_THRESHOLD = 0.10

    @property
    def agent_type(self) -> str:
        return "validator"

    async def _build_prompt(self, input_data: ValidatorInput) -> str:
        """Build the validation prompt."""
        return VALIDATOR_USER_TEMPLATE.format(
            context=input_data.to_prompt_context(),
            component_id=input_data.component_id,
        )

    def _parse_response(
        self,
        response: str,
        input_data: ValidatorInput
    ) -> ValidatorOutput:
        """Parse LLM response into ValidatorOutput."""
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise OutputParseError(f"Failed to parse validation response: {e}")

        try:
            return ValidatorOutput(
                component_id=input_data.component_id,
                validation_result=ValidationResult(data.get("validation_result", "warning")),
                completeness=CompletenessScore(**data.get("completeness", {})),
                call_graph_accuracy=CallGraphAccuracy(**data.get("call_graph_accuracy", {})),
                corrections_applied=[
                    Correction(**c) for c in data.get("corrections_applied", [])
                ],
                corrected_documentation=data.get("corrected_documentation"),
                metadata=data.get("metadata", {}),
            )
        except Exception as e:
            raise OutputParseError(f"Failed to construct validator output: {e}")

    def _validate_output(self, output: ValidatorOutput) -> list[str]:
        """Validate the validator's output structure."""
        errors = []

        # Ensure scores are in valid range
        if not 0 <= output.completeness.score <= 1:
            errors.append("Completeness score out of range")

        if not 0 <= output.call_graph_accuracy.score <= 1:
            errors.append("Call graph accuracy score out of range")

        # Check that corrections have required fields
        for correction in output.corrections_applied:
            if not correction.reason:
                errors.append(f"Correction missing reason: {correction.field}")

        return errors

    def determine_validation_result(
        self,
        completeness: CompletenessScore,
        call_graph: CallGraphAccuracy
    ) -> ValidationResult:
        """
        Determine overall validation result based on scores.

        Rules:
        - FAIL: Any missing parameters OR call graph accuracy < threshold
        - WARNING: Minor issues that don't affect correctness
        - PASS: All checks pass
        """
        # Check for failures
        if completeness.missing_elements:
            return ValidationResult.FAIL

        # Calculate false positive rate for callees
        total_callees = (
            len(call_graph.verified_callees) +
            len(call_graph.false_callees)
        )
        if total_callees > 0:
            false_positive_rate = len(call_graph.false_callees) / total_callees
            if false_positive_rate > self.CALLEE_FALSE_POSITIVE_THRESHOLD:
                return ValidationResult.FAIL

        # Check for warnings
        if call_graph.missing_callees or completeness.extra_elements:
            return ValidationResult.WARNING

        return ValidationResult.PASS


class ClaudeValidatorAgent(ValidatorAgent):
    """Validator agent using Claude Haiku 4.5."""

    def __init__(self, agent_id: str, llm_client: LLMClient, **kwargs):
        model_config = ModelConfig(
            tier=ModelTier.VALIDATION,
            provider="anthropic",
            model_name="claude-haiku-4-5-20251001",
            cost_per_million=0.25,
        )
        super().__init__(
            agent_id=agent_id,
            model_config=model_config,
            llm_client=llm_client,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            **kwargs,
        )


class OpenAIValidatorAgent(ValidatorAgent):
    """Validator agent using GPT-4o-mini."""

    def __init__(self, agent_id: str, llm_client: LLMClient, **kwargs):
        model_config = ModelConfig(
            tier=ModelTier.VALIDATION,
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_million=0.15,
        )
        super().__init__(
            agent_id=agent_id,
            model_config=model_config,
            llm_client=llm_client,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            **kwargs,
        )
```

---

## 6. Documentation Stream Architecture

### 6.1 Stream Class

```python
# twinscribe/streams/documentation_stream.py

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import asyncio

from twinscribe.agents.documenter.agent import DocumenterAgent
from twinscribe.agents.documenter.models import DocumenterInput, DocumenterOutput
from twinscribe.agents.validator.agent import ValidatorAgent
from twinscribe.agents.validator.models import ValidatorInput, ValidatorOutput
from twinscribe.analysis.static_oracle import StaticAnalysisOracle


@dataclass
class StreamResult:
    """Result from processing a single component through the stream."""
    component_id: str
    documentation: DocumenterOutput
    validation: ValidatorOutput
    processing_time_ms: int
    iteration: int


@dataclass
class StreamState:
    """Current state of the documentation stream."""
    stream_id: str
    components_processed: int = 0
    components_total: int = 0
    current_component: Optional[str] = None
    corrections_applied: int = 0
    results: dict[str, StreamResult] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DocumentationStream:
    """
    A documentation stream that processes components through documenter -> validator pipeline.

    Each stream operates independently with its own model configuration.
    Stream A uses Claude models, Stream B uses OpenAI models.

    Flow:
    1. Documenter agent generates documentation
    2. Validator agent validates against source and static analysis
    3. If validation fails, corrections are applied
    4. Results are stored for comparison

    The stream processes components in topological order, providing
    documentation of dependencies as context for better accuracy.
    """

    def __init__(
        self,
        stream_id: str,
        documenter: DocumenterAgent,
        validator: ValidatorAgent,
        static_oracle: StaticAnalysisOracle,
    ):
        self.stream_id = stream_id
        self.documenter = documenter
        self.validator = validator
        self.static_oracle = static_oracle
        self.state = StreamState(stream_id=stream_id)
        self._documented: dict[str, DocumenterOutput] = {}

    async def process(
        self,
        components: list[dict],  # Components in topological order
        iteration: int = 1,
    ) -> dict[str, StreamResult]:
        """
        Process all components through the documentation stream.

        Args:
            components: List of component dicts in topological order
            iteration: Current iteration number (for tracking)

        Returns:
            Dict mapping component_id to StreamResult
        """
        self.state.started_at = datetime.now()
        self.state.components_total = len(components)

        results: dict[str, StreamResult] = {}

        for i, component in enumerate(components):
            component_id = component["component_id"]
            self.state.current_component = component_id
            self.state.components_processed = i + 1

            result = await self._process_component(component, iteration)
            results[component_id] = result
            self.state.results[component_id] = result

            # Store documentation for use as dependency context
            self._documented[component_id] = result.documentation

        self.state.completed_at = datetime.now()
        self.state.current_component = None

        return results

    async def _process_component(
        self,
        component: dict,
        iteration: int
    ) -> StreamResult:
        """Process a single component through documenter -> validator."""
        start_time = datetime.now()
        component_id = component["component_id"]

        # Build dependency context from already-documented components
        dependency_context = self._build_dependency_context(component)

        # Get static analysis hints
        static_hints = self.static_oracle.get_call_graph_for_component(component_id)

        # Step 1: Document
        doc_input = DocumenterInput(
            component_id=component_id,
            source_code=component["source_code"],
            file_path=component["file_path"],
            line_start=component["line_start"],
            line_end=component["line_end"],
            component_type=component["component_type"],
            dependency_context=dependency_context,
            static_analysis_hints=static_hints,
        )

        doc_output, doc_metadata = await self.documenter.process(doc_input)

        # Step 2: Validate
        val_input = ValidatorInput(
            component_id=component_id,
            documentation_output=doc_output.to_dict(),
            source_code=component["source_code"],
            static_call_graph=static_hints.model_dump() if static_hints else {},
        )

        val_output, val_metadata = await self.validator.process(val_input)

        # Apply corrections if needed
        final_doc = doc_output
        if val_output.corrected_documentation:
            final_doc = DocumenterOutput(**val_output.corrected_documentation)
            self.state.corrections_applied += 1

        end_time = datetime.now()
        processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return StreamResult(
            component_id=component_id,
            documentation=final_doc,
            validation=val_output,
            processing_time_ms=processing_time_ms,
            iteration=iteration,
        )

    def _build_dependency_context(
        self,
        component: dict
    ) -> dict[str, DocumentationContent]:
        """Build dependency context from already-documented components."""
        context = {}

        for dep_id in component.get("dependencies", []):
            if dep_id in self._documented:
                context[dep_id] = self._documented[dep_id].documentation

        return context

    async def apply_correction(
        self,
        component_id: str,
        corrected_value: dict
    ) -> None:
        """
        Apply an external correction to a component's documentation.

        This is called when the comparator or Beads resolution provides
        a correction that should override the current documentation.
        """
        if component_id in self._documented:
            # Create new DocumenterOutput with corrected values
            current = self._documented[component_id]
            corrected = DocumenterOutput(
                component_id=component_id,
                documentation=DocumentationContent(**corrected_value.get(
                    "documentation",
                    current.documentation.model_dump()
                )),
                call_graph=CallGraph(**corrected_value.get(
                    "call_graph",
                    current.call_graph.model_dump()
                )),
                metadata={
                    **current.metadata,
                    "correction_applied": True,
                    "correction_source": "external",
                },
            )
            self._documented[component_id] = corrected
            self.state.corrections_applied += 1

    def get_output(self, component_id: str) -> Optional[DocumenterOutput]:
        """Get the current documentation output for a component."""
        return self._documented.get(component_id)

    def get_all_outputs(self) -> dict[str, DocumenterOutput]:
        """Get all documentation outputs."""
        return self._documented.copy()

    def get_statistics(self) -> dict:
        """Get stream processing statistics."""
        return {
            "stream_id": self.stream_id,
            "components_processed": self.state.components_processed,
            "components_total": self.state.components_total,
            "corrections_applied": self.state.corrections_applied,
            "documenter_stats": self.documenter.get_statistics(),
            "validator_stats": self.validator.get_statistics(),
        }
```

---

## 7. System Prompt Templates

```python
# twinscribe/agents/prompts.py

DOCUMENTER_SYSTEM_PROMPT = """You are a code documentation agent. Your task is to generate comprehensive
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

OUTPUT FORMAT:
Respond with a JSON object matching this schema:
{
  "documentation": {
    "summary": "One-line description (required)",
    "description": "Detailed explanation (required)",
    "parameters": [{"name": "...", "type": "...", "description": "...", "default": null}],
    "returns": {"type": "...", "description": "..."},
    "raises": [{"type": "...", "condition": "..."}],
    "examples": ["usage_example()"],
    "notes": "Additional notes (optional)",
    "see_also": ["related.function"]
  },
  "call_graph": {
    "callers": [{"component_id": "...", "call_site_line": 0, "call_type": "direct"}],
    "callees": [{"component_id": "...", "call_site_line": 0, "call_type": "direct"}]
  },
  "metadata": {
    "confidence": 0.85
  }
}

IMPORTANT:
- Do not hallucinate call relationships that don't exist in the code
- If uncertain about a call, still include it but lower your confidence
- Use dependency context to improve accuracy of descriptions
"""

DOCUMENTER_USER_TEMPLATE = """Please document the following {component_type}:

{context}

Component ID to document: {component_id}

Generate comprehensive documentation with accurate call graph linkages.
Respond with valid JSON only.
"""

VALIDATOR_SYSTEM_PROMPT = """You are a documentation validation agent. Your task is to verify that
documentation is complete and that call graph linkages are accurate.

You have access to:
1. The documentation to validate
2. The original source code
3. Static analysis call graph (THIS IS GROUND TRUTH)

VALIDATION RULES:
- All parameters in code must be documented
- All return paths must be documented
- All exceptions that can be raised must be documented
- Call graph MUST match static analysis (static analysis wins if conflict)

VALIDATION CHECKS:
| Check                    | Method                          | Fail Threshold     |
|--------------------------|--------------------------------|-------------------|
| Parameter completeness   | AST parameter extraction       | Any missing       |
| Return type documented   | AST return analysis            | If returns value  |
| Callees accuracy         | Static analysis comparison     | >5% false positive|
| Callers accuracy         | Static analysis comparison     | >10% missing      |

If you find discrepancies between documented call graph and static analysis:
- Trust static analysis (it is ground truth)
- Flag the discrepancy
- Apply correction in your output

OUTPUT FORMAT:
Respond with a JSON object matching this schema:
{
  "validation_result": "pass|fail|warning",
  "completeness": {
    "score": 0.95,
    "missing_elements": ["description of missing element"],
    "extra_elements": []
  },
  "call_graph_accuracy": {
    "score": 0.98,
    "verified_callees": ["component.id"],
    "missing_callees": [],
    "false_callees": [],
    "verified_callers": ["component.id"],
    "missing_callers": [],
    "false_callers": []
  },
  "corrections_applied": [
    {
      "field": "call_graph.callees",
      "action": "removed|added|modified",
      "value": "affected_value",
      "reason": "explanation"
    }
  ],
  "corrected_documentation": null or {corrected full documentation object},
  "metadata": {
    "static_analyzer": "pycg"
  }
}
"""

VALIDATOR_USER_TEMPLATE = """Please validate the following documentation:

{context}

Component ID: {component_id}

Verify completeness and call graph accuracy against static analysis.
Apply corrections if needed. Respond with valid JSON only.
"""
```

---

## 8. Error Handling

```python
# twinscribe/agents/exceptions.py

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class AgentProcessingError(AgentError):
    """Raised when agent processing fails after retries."""
    pass


class OutputParseError(AgentError):
    """Raised when agent output cannot be parsed."""
    pass


class OutputValidationError(AgentError):
    """Raised when agent output fails validation."""
    pass


class StreamError(Exception):
    """Base exception for stream-related errors."""
    pass


class ComponentNotFoundError(StreamError):
    """Raised when a component cannot be found."""
    pass


class DependencyContextError(StreamError):
    """Raised when dependency context cannot be built."""
    pass
```

---

## 9. Configuration Patterns

```python
# twinscribe/config/agents.py

from enum import Enum
from pydantic import BaseModel, Field


class ModelTier(str, Enum):
    """Model pricing/capability tier."""
    GENERATION = "generation"      # ~$3/M tokens
    VALIDATION = "validation"      # ~$0.25/M tokens
    ARBITRATION = "arbitration"    # ~$15/M tokens


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    tier: ModelTier
    provider: str  # "anthropic" or "openai"
    model_name: str
    cost_per_million: float
    max_tokens: int = 4096
    temperature: float = 0.3


class StreamConfig(BaseModel):
    """Configuration for a documentation stream."""
    stream_id: str
    documenter_model: ModelConfig
    validator_model: ModelConfig
    max_retries: int = 3
    timeout_seconds: int = 60


class AgentSystemConfig(BaseModel):
    """Top-level agent system configuration."""
    stream_a: StreamConfig = Field(
        default_factory=lambda: StreamConfig(
            stream_id="A",
            documenter_model=ModelConfig(
                tier=ModelTier.GENERATION,
                provider="anthropic",
                model_name="claude-sonnet-4-5-20250929",
                cost_per_million=3.0,
            ),
            validator_model=ModelConfig(
                tier=ModelTier.VALIDATION,
                provider="anthropic",
                model_name="claude-haiku-4-5-20251001",
                cost_per_million=0.25,
            ),
        )
    )
    stream_b: StreamConfig = Field(
        default_factory=lambda: StreamConfig(
            stream_id="B",
            documenter_model=ModelConfig(
                tier=ModelTier.GENERATION,
                provider="openai",
                model_name="gpt-4o",
                cost_per_million=2.5,
            ),
            validator_model=ModelConfig(
                tier=ModelTier.VALIDATION,
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_million=0.15,
            ),
        )
    )
```

---

## 10. Data Flow Diagram

```
Component Discovery
        |
        v
+-------+--------+
| Topological    |
| Sort           |
+-------+--------+
        |
        v
+-------+--------+                  +----------------+
| For each       |                  | Static         |
| component in   |<-----------------| Analysis       |
| order          |                  | Oracle         |
+-------+--------+                  +----------------+
        |
        +-----------------------------+
        |                             |
        v                             v
+-------+--------+           +-------+--------+
| Stream A       |           | Stream B       |
|                |           |                |
| +------------+ |           | +------------+ |
| | Documenter | |           | | Documenter | |
| | (Claude    | |           | | (GPT-4o)   | |
| | Sonnet)    | |           | |            | |
| +-----+------+ |           | +-----+------+ |
|       |        |           |       |        |
|       v        |           |       v        |
| +------------+ |           | +------------+ |
| | Validator  | |           | | Validator  | |
| | (Claude    | |           | | (GPT-4o-   | |
| | Haiku)     | |           | | mini)      | |
| +-----+------+ |           | +-----+------+ |
|       |        |           |       |        |
+-------+--------+           +-------+--------+
        |                             |
        +-------------+---------------+
                      |
                      v
              +-------+--------+
              | Comparator     |
              | Agent          |
              | (Claude Opus)  |
              +----------------+
```

---

## 11. Interface Contracts Summary

| Interface | Input Type | Output Type | Error Types |
|-----------|------------|-------------|-------------|
| `DocumenterAgent.process()` | `DocumenterInput` | `(DocumenterOutput, AgentMetadata)` | `AgentProcessingError`, `OutputParseError` |
| `ValidatorAgent.process()` | `ValidatorInput` | `(ValidatorOutput, AgentMetadata)` | `AgentProcessingError`, `OutputParseError` |
| `DocumentationStream.process()` | `list[dict]` | `dict[str, StreamResult]` | `StreamError` |
| `DocumentationStream.apply_correction()` | `str, dict` | `None` | `ComponentNotFoundError` |

---

## 12. Testing Strategy

### Unit Tests
- Test `_build_prompt()` produces valid prompts
- Test `_parse_response()` handles various JSON formats
- Test `_validate_output()` catches invalid outputs
- Mock LLM client for deterministic testing

### Integration Tests
- Test full stream processing with mock components
- Test correction application propagates correctly
- Test statistics collection accuracy

### Contract Tests
- Verify output schemas match spec
- Verify input validation catches malformed data
- Test protocol compliance

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Systems Architect | Initial design |
