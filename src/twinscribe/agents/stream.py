"""
Documentation Stream Interface.

Defines the interface for a complete documentation stream that manages
a documenter/validator pair and processes components in topological order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from twinscribe.agents.documenter import DocumenterAgent, DocumenterConfig
from twinscribe.agents.validator import ValidatorAgent, ValidatorConfig
from twinscribe.models.base import StreamId
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import Component
from twinscribe.models.documentation import DocumentationOutput, StreamOutput
from twinscribe.models.validation import ValidationResult


class StreamConfig(BaseModel):
    """Configuration for a documentation stream.

    Attributes:
        stream_id: Stream identifier (A or B)
        documenter_config: Configuration for the documenter agent
        validator_config: Configuration for the validator agent
        batch_size: Number of components to process in parallel
        max_retries: Maximum retries per component
        continue_on_error: Whether to continue if a component fails
    """

    stream_id: StreamId = Field(..., description="Stream identifier")
    documenter_config: DocumenterConfig = Field(
        ..., description="Documenter agent configuration"
    )
    validator_config: ValidatorConfig = Field(
        ..., description="Validator agent configuration"
    )
    batch_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Parallel processing batch size",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retries per component",
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue if component fails",
    )


@dataclass
class StreamResult:
    """Result from processing a stream.

    Attributes:
        stream_id: Stream identifier
        output: Stream output with all documentation
        successful: Number of successfully processed components
        failed: Number of failed components
        failed_component_ids: IDs of components that failed
        total_tokens: Total tokens consumed
        total_cost: Total cost in USD
        started_at: Processing start time
        completed_at: Processing end time
    """

    stream_id: StreamId
    output: StreamOutput
    successful: int = 0
    failed: int = 0
    failed_component_ids: list[str] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Total processing duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Percentage of components successfully processed."""
        total = self.successful + self.failed
        return self.successful / total if total > 0 else 0.0


@dataclass
class ComponentProcessingResult:
    """Result from processing a single component.

    Attributes:
        component_id: Component that was processed
        documentation: Documentation output (if successful)
        validation: Validation result (if successful)
        success: Whether processing succeeded
        error: Error message if failed
        retries: Number of retries needed
    """

    component_id: str
    documentation: Optional[DocumentationOutput] = None
    validation: Optional[ValidationResult] = None
    success: bool = True
    error: Optional[str] = None
    retries: int = 0


class DocumentationStream(ABC):
    """Abstract base class for a documentation stream.

    A stream manages a documenter/validator pair and processes
    components in topological order. Two streams (A and B) run
    in parallel with different models.

    The stream:
    1. Receives components in topological order
    2. For each component:
       a. Calls documenter to generate documentation
       b. Calls validator to verify against ground truth
       c. Applies corrections if needed
    3. Produces complete StreamOutput

    Lifecycle:
    1. initialize() - Set up agents
    2. process() - Process all components
    3. apply_correction() - Apply corrections from comparison
    4. shutdown() - Clean up resources
    """

    def __init__(self, config: StreamConfig) -> None:
        """Initialize the documentation stream.

        Args:
            config: Stream configuration
        """
        self._config = config
        self._documenter: Optional[DocumenterAgent] = None
        self._validator: Optional[ValidatorAgent] = None
        self._output = StreamOutput(stream_id=config.stream_id)
        self._initialized = False

    @property
    def config(self) -> StreamConfig:
        """Get stream configuration."""
        return self._config

    @property
    def stream_id(self) -> StreamId:
        """Get stream identifier."""
        return self._config.stream_id

    @property
    def output(self) -> StreamOutput:
        """Get current stream output."""
        return self._output

    @property
    def is_initialized(self) -> bool:
        """Check if stream is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the stream and its agents.

        Creates and initializes the documenter and validator agents.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def process(
        self,
        components: list[Component],
        source_code_map: dict[str, str],
        ground_truth: CallGraph,
    ) -> StreamResult:
        """Process all components through the stream.

        Processes components in order, building up dependency context
        as earlier components are documented.

        Args:
            components: Components in topological order
            source_code_map: Map of component_id -> source code
            ground_truth: Static analysis call graph

        Returns:
            StreamResult with all documentation and metrics

        Raises:
            RuntimeError: If stream not initialized
        """
        pass

    @abstractmethod
    async def process_component(
        self,
        component: Component,
        source_code: str,
        dependency_context: dict[str, DocumentationOutput],
        ground_truth: CallGraph,
    ) -> ComponentProcessingResult:
        """Process a single component.

        Runs the component through documenter then validator.

        Args:
            component: Component to process
            source_code: Component source code
            dependency_context: Documentation of dependencies
            ground_truth: Static analysis call graph

        Returns:
            Processing result for this component
        """
        pass

    @abstractmethod
    async def apply_correction(
        self,
        component_id: str,
        corrected_value: any,
        field_path: str,
    ) -> bool:
        """Apply a correction to a component's documentation.

        Called by the comparator when ground truth or human review
        indicates a correction is needed.

        Args:
            component_id: Component to correct
            corrected_value: New value to apply
            field_path: Path to field being corrected

        Returns:
            True if correction was applied

        Raises:
            ValueError: If component not found
        """
        pass

    @abstractmethod
    async def reprocess_component(
        self,
        component_id: str,
        corrections: list[dict],
    ) -> ComponentProcessingResult:
        """Reprocess a component with corrections.

        Called when a component needs to be re-documented with
        specific corrections applied.

        Args:
            component_id: Component to reprocess
            corrections: List of corrections to apply

        Returns:
            New processing result
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the stream and its agents.

        Releases all resources and cleans up.
        """
        pass

    def get_documentation(
        self, component_id: str
    ) -> Optional[DocumentationOutput]:
        """Get documentation for a specific component.

        Args:
            component_id: Component to look up

        Returns:
            Documentation output or None if not found
        """
        return self._output.get_output(component_id)

    def get_all_component_ids(self) -> list[str]:
        """Get IDs of all documented components.

        Returns:
            List of component IDs
        """
        return list(self._output.outputs.keys())

    def reset(self) -> None:
        """Reset stream output for a new iteration.

        Clears all stored documentation for re-processing.
        """
        self._output = StreamOutput(stream_id=self._config.stream_id)


class StreamProgressCallback:
    """Callback interface for stream progress updates.

    Implement this to receive progress notifications during
    stream processing.
    """

    async def on_component_start(
        self,
        stream_id: StreamId,
        component_id: str,
        index: int,
        total: int,
    ) -> None:
        """Called when starting to process a component.

        Args:
            stream_id: Stream identifier
            component_id: Component being processed
            index: Current index (0-based)
            total: Total components
        """
        pass

    async def on_component_complete(
        self,
        stream_id: StreamId,
        component_id: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Called when component processing completes.

        Args:
            stream_id: Stream identifier
            component_id: Component that was processed
            success: Whether processing succeeded
            duration_ms: Processing duration
        """
        pass

    async def on_validation_complete(
        self,
        stream_id: StreamId,
        component_id: str,
        passed: bool,
        corrections: int,
    ) -> None:
        """Called when validation completes.

        Args:
            stream_id: Stream identifier
            component_id: Component that was validated
            passed: Whether validation passed
            corrections: Number of corrections applied
        """
        pass

    async def on_stream_complete(
        self,
        stream_id: StreamId,
        result: StreamResult,
    ) -> None:
        """Called when stream processing completes.

        Args:
            stream_id: Stream identifier
            result: Final stream result
        """
        pass

    async def on_error(
        self,
        stream_id: StreamId,
        component_id: str,
        error: Exception,
    ) -> None:
        """Called when an error occurs.

        Args:
            stream_id: Stream identifier
            component_id: Component that failed
            error: The exception that occurred
        """
        pass
