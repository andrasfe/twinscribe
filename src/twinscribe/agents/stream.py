"""
Documentation Stream Interface.

Defines the interface for a complete documentation stream that manages
a documenter/validator pair and processes components in topological order.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from twinscribe.orchestrator.checkpoint import CheckpointManager

from pydantic import BaseModel, Field

from twinscribe.agents.documenter import (
    DocumenterAgent,
    DocumenterConfig,
    DocumenterInput,
)
from twinscribe.agents.validator import (
    ValidatorAgent,
    ValidatorConfig,
    ValidatorInput,
)
from twinscribe.models.base import CallType, StreamId, ValidationStatus
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ExceptionDoc,
    ParameterDoc,
    ReturnDoc,
)
from twinscribe.models.documentation import (
    CalleeRef,
    CallerRef,
    CallGraphSection,
    DocumentationOutput,
    DocumenterMetadata,
    StreamOutput,
)
from twinscribe.models.validation import (
    CallGraphAccuracy,
    CompletenessCheck,
    CorrectionApplied,
    ValidationResult,
    ValidatorMetadata,
)
from twinscribe.utils.llm_client import AsyncLLMClient, Message


class StreamConfig(BaseModel):
    """Configuration for a documentation stream.

    Attributes:
        stream_id: Stream identifier (A or B)
        documenter_config: Configuration for the documenter agent
        validator_config: Configuration for the validator agent
        batch_size: Number of components to process in parallel
        max_retries: Maximum retries per component
        continue_on_error: Whether to continue if a component fails
        rate_limit_delay: Delay in seconds between API calls to avoid rate limits
    """

    stream_id: StreamId = Field(..., description="Stream identifier")
    documenter_config: DocumenterConfig = Field(..., description="Documenter agent configuration")
    validator_config: ValidatorConfig = Field(..., description="Validator agent configuration")
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
        default=False,
        description="Continue if component fails (fail-fast by default)",
    )
    rate_limit_delay: float = Field(
        default=1.0,
        ge=0,
        description="Delay between API calls in seconds",
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
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
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
    documentation: DocumentationOutput | None = None
    validation: ValidationResult | None = None
    success: bool = True
    error: str | None = None
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
        self._documenter: DocumenterAgent | None = None
        self._validator: ValidatorAgent | None = None
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

    def get_documentation(self, component_id: str) -> DocumentationOutput | None:
        """Get documentation for a specific component.

        Args:
            component_id: Component to look up

        Returns:
            Documentation output or None if not found
        """
        return self._output.get_output(component_id)

    def get_outputs(self) -> dict[str, DocumentationOutput]:
        """Get all documentation outputs.

        Returns:
            Dict mapping component_id to DocumentationOutput
        """
        return self._output.outputs

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


class ConcreteDocumentationStream(DocumentationStream):
    """Concrete implementation of a documentation stream.

    Manages a documenter/validator pair to process components in
    topological order, generating and validating documentation.

    This implementation:
    1. Creates and manages DocumenterAgent and ValidatorAgent instances
    2. Processes components sequentially with dependency context
    3. Validates generated documentation against ground truth
    4. Supports corrections and reprocessing
    5. Tracks token usage and costs throughout
    6. Records checkpoints after each component via CheckpointManager

    Example:
        config = StreamConfig(
            stream_id=StreamId.STREAM_A,
            documenter_config=STREAM_A_DOCUMENTER_CONFIG,
            validator_config=STREAM_A_VALIDATOR_CONFIG,
        )
        stream = ConcreteDocumentationStream(config)
        await stream.initialize()
        result = await stream.process(components, source_map, ground_truth)
        await stream.shutdown()
    """

    def __init__(
        self,
        config: StreamConfig,
        progress_callback: StreamProgressCallback | None = None,
        checkpoint_manager: "CheckpointManager | None" = None,
    ) -> None:
        """Initialize the concrete documentation stream.

        Args:
            config: Stream configuration with documenter and validator settings
            progress_callback: Optional callback for progress updates
            checkpoint_manager: Optional checkpoint manager for state persistence
        """
        super().__init__(config)
        self._progress_callback = progress_callback
        self._checkpoint_manager = checkpoint_manager
        self._llm_client: AsyncLLMClient | None = None
        self._source_code_map: dict[str, str] = {}
        self._ground_truth: CallGraph | None = None
        self._processing_context: dict[str, DocumentationOutput] = {}
        self._component_map: dict[str, Component] = {}
        self._current_iteration: int = 1
        self._logger = logging.getLogger(f"{__name__}.{config.stream_id.value}")

    async def initialize(self) -> None:
        """Initialize the stream and its agents.

        Creates the LLM client and initializes both the documenter
        and validator agents.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            self._logger.warning("Stream already initialized")
            return

        try:
            self._logger.info(f"Initializing stream {self._config.stream_id.value}")

            # Create LLM client
            self._llm_client = AsyncLLMClient(
                app_title=f"TwinScribe-Stream-{self._config.stream_id.value}",
                max_retries=self._config.max_retries,
            )

            # Initialize documenter agent
            self._documenter = ConcreteDocumenterAgent(
                config=self._config.documenter_config,
                llm_client=self._llm_client,
            )
            await self._documenter.initialize()

            # Initialize validator agent
            self._validator = ConcreteValidatorAgent(
                config=self._config.validator_config,
                llm_client=self._llm_client,
            )
            await self._validator.initialize()

            self._initialized = True
            self._logger.info(f"Stream {self._config.stream_id.value} initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize stream: {e}")
            raise RuntimeError(f"Stream initialization failed: {e}") from e

    async def process(
        self,
        components: list[Component],
        source_code_map: dict[str, str],
        ground_truth: CallGraph,
    ) -> StreamResult:
        """Process all components through the stream.

        Processes components in topological order, building up dependency
        context as earlier components are documented.

        Args:
            components: Components in topological order
            source_code_map: Map of component_id -> source code
            ground_truth: Static analysis call graph

        Returns:
            StreamResult with all documentation and metrics

        Raises:
            RuntimeError: If stream not initialized
        """
        if not self._initialized:
            raise RuntimeError("Stream not initialized. Call initialize() first.")

        self._logger.info(f"Starting to process {len(components)} components")

        # Store references for reprocessing
        self._source_code_map = source_code_map
        self._ground_truth = ground_truth
        self._component_map = {c.component_id: c for c in components}
        self._processing_context = {}

        # Initialize result
        result = StreamResult(
            stream_id=self._config.stream_id,
            output=self._output,
            started_at=datetime.now(),
        )

        total = len(components)
        stream_name = "Stream A" if self._config.stream_id.value == "A" else "Stream B"

        self._logger.info(f"{stream_name}: Processing {total} components")

        for index, component in enumerate(components):
            component_id = component.component_id

            # Log progress
            self._logger.info(
                f"{stream_name}: [{index + 1}/{total}] Processing {component_id}"
            )

            # Notify progress callback
            if self._progress_callback:
                await self._progress_callback.on_component_start(
                    stream_id=self._config.stream_id,
                    component_id=component_id,
                    index=index,
                    total=total,
                )

            start_time = datetime.now()

            try:
                # Get source code
                source_code = source_code_map.get(component_id, "")
                if not source_code:
                    self._logger.warning(f"No source code found for {component_id}")

                # Build dependency context from already processed components
                dependency_context = self._build_dependency_context(component, ground_truth)

                # Process the component
                proc_result = await self.process_component(
                    component=component,
                    source_code=source_code,
                    dependency_context=dependency_context,
                    ground_truth=ground_truth,
                )

                # Update result
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                if proc_result.success:
                    result.successful += 1
                    if proc_result.documentation:
                        self._output.add_output(proc_result.documentation)
                        self._processing_context[component_id] = proc_result.documentation
                        result.total_tokens += proc_result.documentation.metadata.token_count or 0

                        # Record checkpoint after successful documentation
                        if self._checkpoint_manager:
                            token_count = proc_result.documentation.metadata.token_count
                            self._checkpoint_manager.record_component_documented(
                                component_id=component_id,
                                stream_id=self._config.stream_id.value,
                                iteration=self._current_iteration,
                                output=proc_result.documentation,
                                duration_ms=duration_ms,
                                token_count=token_count,
                            )

                    self._logger.info(
                        f"{stream_name}: [{index + 1}/{total}] ✓ {component_id} ({duration_ms:.0f}ms)"
                    )
                else:
                    result.failed += 1
                    result.failed_component_ids.append(component_id)
                    self._logger.warning(
                        f"{stream_name}: [{index + 1}/{total}] ✗ {component_id} failed ({duration_ms:.0f}ms)"
                    )

                    # Record error in checkpoint
                    if self._checkpoint_manager:
                        self._checkpoint_manager.record_error(
                            phase="documenting",
                            component_id=component_id,
                            stream_id=self._config.stream_id.value,
                            error=proc_result.error or "Unknown error",
                        )

                # Notify progress callback
                if self._progress_callback:
                    await self._progress_callback.on_component_complete(
                        stream_id=self._config.stream_id,
                        component_id=component_id,
                        success=proc_result.success,
                        duration_ms=duration_ms,
                    )

                    if proc_result.validation:
                        await self._progress_callback.on_validation_complete(
                            stream_id=self._config.stream_id,
                            component_id=component_id,
                            passed=proc_result.validation.is_valid,
                            corrections=proc_result.validation.total_corrections,
                        )

            except Exception as e:
                self._logger.error(f"Error processing component {component_id}: {e}")
                result.failed += 1
                result.failed_component_ids.append(component_id)

                # Record error in checkpoint BEFORE any potential re-raise
                # This ensures checkpoint is saved for resume capability
                if self._checkpoint_manager:
                    import traceback
                    tb = traceback.format_exc()
                    self._checkpoint_manager.record_error(
                        phase="documenting",
                        component_id=component_id,
                        stream_id=self._config.stream_id.value,
                        error=e,
                        traceback=tb,
                    )

                if self._progress_callback:
                    await self._progress_callback.on_error(
                        stream_id=self._config.stream_id,
                        component_id=component_id,
                        error=e,
                    )

                # Fail-fast: re-raise immediately unless continue_on_error is True
                if not self._config.continue_on_error:
                    raise

            # Rate limit delay between components
            if self._config.rate_limit_delay > 0 and index < total - 1:
                await asyncio.sleep(self._config.rate_limit_delay)

        # Finalize result
        result.completed_at = datetime.now()
        result.output = self._output

        # Get cost from LLM client
        if self._llm_client:
            usage_summary = await self._llm_client.get_usage_summary()
            result.total_cost = usage_summary.get("totals", {}).get("cost_usd", 0.0)

        # Notify completion
        if self._progress_callback:
            await self._progress_callback.on_stream_complete(
                stream_id=self._config.stream_id,
                result=result,
            )

        self._logger.info(
            f"Stream processing complete. Success: {result.successful}, Failed: {result.failed}"
        )

        return result

    async def process_component(
        self,
        component: Component,
        source_code: str,
        dependency_context: dict[str, DocumentationOutput],
        ground_truth: CallGraph,
    ) -> ComponentProcessingResult:
        """Process a single component through documenter and validator.

        Args:
            component: Component to process
            source_code: Component source code
            dependency_context: Documentation of dependencies
            ground_truth: Static analysis call graph

        Returns:
            Processing result for this component
        """
        component_id = component.component_id
        result = ComponentProcessingResult(component_id=component_id)
        retries = 0
        max_retries = self._config.max_retries

        while retries <= max_retries:
            try:
                # Step 1: Generate documentation
                self._logger.debug(f"Documenting component {component_id} (attempt {retries + 1})")

                documenter_input = DocumenterInput(
                    component=component,
                    source_code=source_code,
                    dependency_context=dependency_context,
                    static_analysis_hints=ground_truth,
                    iteration=retries + 1,
                    previous_output=result.documentation,
                )

                documentation = await self._documenter.process(documenter_input)
                result.documentation = documentation

                # Rate limit delay between documenter and validator
                if self._config.rate_limit_delay > 0:
                    await asyncio.sleep(self._config.rate_limit_delay)

                # Step 2: Validate documentation
                self._logger.debug(f"Validating component {component_id}")

                validator_input = ValidatorInput(
                    documentation=documentation,
                    source_code=source_code,
                    ground_truth_call_graph=ground_truth,
                )

                validation = await self._validator.process(validator_input)
                result.validation = validation

                # Check if validation passed
                if validation.is_valid:
                    result.success = True
                    result.retries = retries
                    return result

                # If validation failed but we have corrections, apply them
                if validation.corrections_applied and retries < max_retries:
                    self._logger.debug(
                        f"Applying {len(validation.corrections_applied)} corrections"
                    )
                    # Update documenter input with corrections for next attempt
                    documenter_input.corrections = [
                        {
                            "field": c.field,
                            "action": c.action,
                            "reason": c.reason,
                        }
                        for c in validation.corrections_applied
                    ]
                    retries += 1
                    continue

                # Validation failed but no more retries
                if validation.validation_result == ValidationStatus.FAIL:
                    result.success = False
                    result.error = "Validation failed after corrections"
                else:
                    # WARNING status is acceptable
                    result.success = True

                result.retries = retries
                return result

            except Exception as e:
                self._logger.warning(
                    f"Error processing {component_id} (attempt {retries + 1}): {e}"
                )
                retries += 1
                result.error = str(e)

                if retries > max_retries:
                    result.success = False
                    result.retries = retries - 1
                    return result

        result.success = False
        result.retries = max_retries
        return result

    async def apply_correction(
        self,
        component_id: str,
        corrected_value: any,
        field_path: str,
    ) -> bool:
        """Apply a correction to a component's documentation.

        Updates the stored documentation with the corrected value.

        Args:
            component_id: Component to correct
            corrected_value: New value to apply
            field_path: Path to field being corrected (e.g., "call_graph.callees")

        Returns:
            True if correction was applied

        Raises:
            ValueError: If component not found
        """
        documentation = self._output.get_output(component_id)
        if documentation is None:
            raise ValueError(f"Component {component_id} not found in output")

        try:
            # Parse field path and apply correction
            parts = field_path.split(".")
            obj = documentation

            # Navigate to parent of target field
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict):
                    obj = obj[part]
                else:
                    raise ValueError(f"Invalid field path: {field_path}")

            # Apply correction to final field
            final_field = parts[-1]
            if hasattr(obj, final_field):
                # For Pydantic models, we need to handle this carefully
                # Create a dict representation, modify it, and recreate
                obj_dict = obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
                obj_dict[final_field] = corrected_value

                # Update the field - this is a simplified approach
                # In practice, you might need to recreate the entire object
                setattr(obj, final_field, corrected_value)
            elif isinstance(obj, dict):
                obj[final_field] = corrected_value
            else:
                raise ValueError(f"Cannot set field: {final_field}")

            # Update the processing context as well
            self._processing_context[component_id] = documentation

            self._logger.info(f"Applied correction to {component_id}.{field_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to apply correction to {component_id}: {e}")
            return False

    async def reprocess_component(
        self,
        component_id: str,
        corrections: list[dict],
    ) -> ComponentProcessingResult:
        """Reprocess a component with corrections.

        Args:
            component_id: Component to reprocess
            corrections: List of corrections to apply

        Returns:
            New processing result

        Raises:
            ValueError: If component not found or missing required data
        """
        # Get component from stored map
        component = self._component_map.get(component_id)
        if component is None:
            raise ValueError(f"Component {component_id} not found")

        # Get source code
        source_code = self._source_code_map.get(component_id, "")
        if not source_code:
            raise ValueError(f"No source code found for {component_id}")

        # Get ground truth
        if self._ground_truth is None:
            raise ValueError("Ground truth not available")

        # Build dependency context
        dependency_context = self._build_dependency_context(component, self._ground_truth)

        # Get previous documentation if available
        previous_output = self._output.get_output(component_id)

        # Create documenter input with corrections
        documenter_input = DocumenterInput(
            component=component,
            source_code=source_code,
            dependency_context=dependency_context,
            static_analysis_hints=self._ground_truth,
            iteration=2,  # Mark as reprocessing
            previous_output=previous_output,
            corrections=corrections,
        )

        # Process through documenter
        try:
            documentation = await self._documenter.process(documenter_input)

            # Rate limit delay between documenter and validator
            if self._config.rate_limit_delay > 0:
                await asyncio.sleep(self._config.rate_limit_delay)

            # Validate
            validator_input = ValidatorInput(
                documentation=documentation,
                source_code=source_code,
                ground_truth_call_graph=self._ground_truth,
            )
            validation = await self._validator.process(validator_input)

            # Update stored output
            self._output.add_output(documentation)
            self._processing_context[component_id] = documentation

            return ComponentProcessingResult(
                component_id=component_id,
                documentation=documentation,
                validation=validation,
                success=validation.is_valid,
                retries=0,
            )

        except Exception as e:
            self._logger.error(f"Failed to reprocess {component_id}: {e}")
            return ComponentProcessingResult(
                component_id=component_id,
                success=False,
                error=str(e),
            )

    async def shutdown(self) -> None:
        """Shutdown the stream and its agents.

        Releases all resources including LLM client and agents.
        """
        self._logger.info(f"Shutting down stream {self._config.stream_id.value}")

        if self._documenter:
            await self._documenter.shutdown()
            self._documenter = None

        if self._validator:
            await self._validator.shutdown()
            self._validator = None

        if self._llm_client:
            await self._llm_client.close()
            self._llm_client = None

        self._initialized = False
        self._processing_context.clear()
        self._component_map.clear()
        self._source_code_map.clear()
        self._ground_truth = None

        self._logger.info("Stream shutdown complete")

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration number.

        Called by the orchestrator at the start of each iteration.

        Args:
            iteration: Current iteration number (1-based)
        """
        self._current_iteration = iteration

    def set_checkpoint_manager(self, manager: "CheckpointManager") -> None:
        """Set the checkpoint manager for state persistence.

        Args:
            manager: CheckpointManager instance
        """
        self._checkpoint_manager = manager

    @property
    def checkpoint_manager(self) -> "CheckpointManager | None":
        """Get the checkpoint manager if set."""
        return self._checkpoint_manager

    def _build_dependency_context(
        self,
        component: Component,
        ground_truth: CallGraph,
    ) -> dict[str, DocumentationOutput]:
        """Build dependency context from already processed components.

        Gets documentation for components that this component calls,
        which have already been processed.

        Args:
            component: The component being processed
            ground_truth: Call graph for dependency lookup

        Returns:
            Dict mapping dependency component_id to their documentation
        """
        context = {}

        # Get callees from ground truth
        callees = ground_truth.get_callees(component.component_id)

        for edge in callees:
            callee_id = edge.callee
            # Check if we've already documented this component
            if callee_id in self._processing_context:
                context[callee_id] = self._processing_context[callee_id]

        return context


class ConcreteDocumenterAgent(DocumenterAgent):
    """Concrete implementation of the documenter agent.

    Uses the AsyncLLMClient to generate documentation for code components.
    """

    def __init__(
        self,
        config: DocumenterConfig,
        llm_client: AsyncLLMClient | None = None,
    ) -> None:
        """Initialize the concrete documenter agent.

        Args:
            config: Documenter configuration
            llm_client: Optional shared LLM client
        """
        super().__init__(config)
        self._llm_client = llm_client
        self._owns_client = llm_client is None
        self._logger = logging.getLogger(f"{__name__}.Documenter.{config.agent_id}")

    async def initialize(self) -> None:
        """Initialize the documenter agent."""
        if self._initialized:
            return

        if self._llm_client is None:
            self._llm_client = AsyncLLMClient(
                app_title=f"TwinScribe-Documenter-{self._config.agent_id}",
            )
            self._owns_client = True

        self._initialized = True
        self._logger.info(f"Documenter agent {self._config.agent_id} initialized")

    async def process(self, input_data: DocumenterInput) -> DocumentationOutput:
        """Generate documentation for a component.

        Args:
            input_data: Input containing component, source, and context

        Returns:
            Structured documentation with call graph

        Raises:
            RuntimeError: If agent not initialized
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        self._metrics.started_at = self._metrics.started_at or datetime.now()

        try:
            # Build prompts
            user_prompt = self._build_user_prompt(input_data)

            # Call LLM
            response = await self._llm_client.generate_documentation(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self._config.model_name,
                json_mode=True,
                max_tokens=self._config.max_tokens,
            )

            # Record metrics
            cost = self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self._metrics.record_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=response.latency_ms,
                cost=cost,
            )

            # Parse response
            output = self._parse_response(
                response.content,
                input_data.component.component_id,
                response.usage.total_tokens,
            )

            return output

        except Exception as e:
            self._metrics.record_error()
            self._logger.error(f"Error documenting {input_data.component.component_id}: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the documenter agent."""
        if self._owns_client and self._llm_client:
            await self._llm_client.close()
            self._llm_client = None
        self._initialized = False
        self._logger.info(f"Documenter agent {self._config.agent_id} shutdown")

    def _parse_response(
        self,
        content: str,
        component_id: str,
        token_count: int,
    ) -> DocumentationOutput:
        """Parse LLM response into DocumentationOutput.

        Args:
            content: JSON response content
            component_id: Component being documented
            token_count: Tokens used

        Returns:
            Parsed DocumentationOutput
        """
        import json
        import re

        # Strip markdown code fences if present (models sometimes wrap JSON in ```json ... ```)
        content = content.strip()
        if content.startswith("```"):
            # Remove opening fence (with optional language specifier)
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            # Remove closing fence
            content = re.sub(r"\n?```\s*$", "", content)
            content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Log detailed error info for debugging
            content_preview = content[:200] if content else "<empty>"
            self._logger.error(
                f"Failed to parse JSON response for {component_id}: {e}\n"
                f"Content length: {len(content) if content else 0}, "
                f"Preview: {content_preview!r}"
            )
            # Return minimal valid output
            return self._create_fallback_output(component_id, token_count)

        # Handle case where model returns a list instead of dict
        if isinstance(data, list):
            # Use first element if it's a dict, otherwise create fallback
            if data and isinstance(data[0], dict):
                self._logger.warning(
                    f"Model returned list instead of dict for {component_id}, using first element"
                )
                data = data[0]
            else:
                self._logger.error(
                    f"Model returned invalid list for {component_id}: {data[:100] if data else '[]'}"
                )
                return self._create_fallback_output(component_id, token_count)

        # Extract documentation section
        doc_data = data.get("documentation", {})
        documentation = ComponentDocumentation(
            summary=doc_data.get("summary", ""),
            description=doc_data.get("description", ""),
            parameters=[
                ParameterDoc(
                    name=p.get("name", ""),
                    type=p.get("type"),
                    description=p.get("description", ""),
                    default=p.get("default"),
                    required=p.get("required", True),
                )
                for p in doc_data.get("parameters", [])
            ],
            returns=ReturnDoc(
                type=doc_data.get("returns", {}).get("type"),
                description=doc_data.get("returns", {}).get("description", ""),
            )
            if doc_data.get("returns")
            else None,
            raises=[
                ExceptionDoc(
                    type=e.get("type", "Exception"),
                    condition=e.get("condition", ""),
                )
                for e in doc_data.get("raises", [])
            ],
            examples=doc_data.get("examples", []),
        )

        # Extract call graph section
        cg_data = data.get("call_graph", {})

        # Filter out entries with empty component_id to avoid Pydantic validation errors
        callers_data = [
            c for c in cg_data.get("callers", [])
            if c.get("component_id", "").strip()
        ]
        callees_data = [
            c for c in cg_data.get("callees", [])
            if c.get("component_id", "").strip()
        ]

        call_graph = CallGraphSection(
            callers=[
                CallerRef(
                    component_id=c.get("component_id", ""),
                    call_site_line=c.get("call_site_line"),
                    call_type=CallType(c.get("call_type", "direct")),
                )
                for c in callers_data
            ],
            callees=[
                CalleeRef(
                    component_id=c.get("component_id", ""),
                    call_site_line=c.get("call_site_line"),
                    call_type=CallType(c.get("call_type", "direct")),
                )
                for c in callees_data
            ],
        )

        # Create metadata
        metadata = DocumenterMetadata(
            agent_id=self._config.agent_id,
            stream_id=self._config.stream_id,
            model=self._config.model_name,
            confidence=data.get("confidence", 0.8),
            token_count=token_count,
        )

        return DocumentationOutput(
            component_id=component_id,
            documentation=documentation,
            call_graph=call_graph,
            metadata=metadata,
        )

    def _create_fallback_output(
        self,
        component_id: str,
        token_count: int,
    ) -> DocumentationOutput:
        """Create a minimal fallback output when parsing fails."""
        return DocumentationOutput(
            component_id=component_id,
            documentation=ComponentDocumentation(
                summary="Documentation generation failed",
                description="Unable to parse LLM response",
            ),
            call_graph=CallGraphSection(),
            metadata=DocumenterMetadata(
                agent_id=self._config.agent_id,
                stream_id=self._config.stream_id,
                model=self._config.model_name,
                confidence=0.0,
                token_count=token_count,
            ),
        )


class ConcreteValidatorAgent(ValidatorAgent):
    """Concrete implementation of the validator agent.

    Uses the AsyncLLMClient to validate documentation against ground truth.
    """

    def __init__(
        self,
        config: ValidatorConfig,
        llm_client: AsyncLLMClient | None = None,
    ) -> None:
        """Initialize the concrete validator agent.

        Args:
            config: Validator configuration
            llm_client: Optional shared LLM client
        """
        super().__init__(config)
        self._llm_client = llm_client
        self._owns_client = llm_client is None
        self._logger = logging.getLogger(f"{__name__}.Validator.{config.agent_id}")

    async def initialize(self) -> None:
        """Initialize the validator agent."""
        if self._initialized:
            return

        if self._llm_client is None:
            self._llm_client = AsyncLLMClient(
                app_title=f"TwinScribe-Validator-{self._config.agent_id}",
            )
            self._owns_client = True

        self._initialized = True
        self._logger.info(f"Validator agent {self._config.agent_id} initialized")

    async def process(self, input_data: ValidatorInput) -> ValidationResult:
        """Validate documentation for a component.

        Args:
            input_data: Input containing documentation and ground truth

        Returns:
            Validation result with corrections

        Raises:
            RuntimeError: If agent not initialized
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized")

        self._metrics.started_at = self._metrics.started_at or datetime.now()

        try:
            # Build prompts
            user_prompt = self._build_user_prompt(input_data)

            # Call LLM
            response = await self._llm_client.send_message(
                model=self._config.model_name,
                messages=[
                    Message(role="system", content=self.SYSTEM_PROMPT),
                    Message(role="user", content=user_prompt),
                ],
                json_mode=True,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            # Record metrics
            cost = self._calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            self._metrics.record_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=response.latency_ms,
                cost=cost,
            )

            # Parse response
            result = self._parse_response(
                response.content,
                input_data.documentation.component_id,
                response.usage.total_tokens,
            )

            return result

        except Exception as e:
            self._metrics.record_error()
            self._logger.error(f"Error validating {input_data.documentation.component_id}: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the validator agent."""
        if self._owns_client and self._llm_client:
            await self._llm_client.close()
            self._llm_client = None
        self._initialized = False
        self._logger.info(f"Validator agent {self._config.agent_id} shutdown")

    def _parse_response(
        self,
        content: str,
        component_id: str,
        token_count: int,
    ) -> ValidationResult:
        """Parse LLM response into ValidationResult.

        Args:
            content: JSON response content
            component_id: Component being validated
            token_count: Tokens used

        Returns:
            Parsed ValidationResult
        """
        import json
        import re

        # Strip markdown code fences if present (models sometimes wrap JSON in ```json ... ```)
        content = content.strip()
        if content.startswith("```"):
            # Remove opening fence (with optional language specifier)
            content = re.sub(r"^```(?:json)?\s*\n?", "", content)
            # Remove closing fence
            content = re.sub(r"\n?```\s*$", "", content)
            content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Log detailed error info for debugging
            content_preview = content[:200] if content else "<empty>"
            self._logger.error(
                f"Failed to parse JSON response for validation of {component_id}: {e}\n"
                f"Content length: {len(content) if content else 0}, "
                f"Preview: {content_preview!r}"
            )
            return self._create_fallback_result(component_id, token_count)

        # Handle case where model returns a list instead of dict
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                self._logger.warning(
                    f"Model returned list instead of dict for validation of {component_id}, using first element"
                )
                data = data[0]
            else:
                self._logger.error(
                    f"Model returned invalid list for validation of {component_id}"
                )
                return self._create_fallback_result(component_id, token_count)

        # Parse validation status
        status_str = data.get("validation_result", "warning")
        try:
            status = ValidationStatus(status_str)
        except ValueError:
            status = ValidationStatus.WARNING

        # Parse completeness check
        comp_data = data.get("completeness", {})
        completeness = CompletenessCheck(
            score=comp_data.get("score", 1.0),
            missing_elements=comp_data.get("missing_elements", []),
            extra_elements=comp_data.get("extra_elements", []),
        )

        # Parse call graph accuracy
        cg_data = data.get("call_graph_accuracy", {})
        call_graph_accuracy = CallGraphAccuracy(
            score=cg_data.get("score", 1.0),
            verified_callees=cg_data.get("verified_callees", []),
            missing_callees=cg_data.get("missing_callees", []),
            false_callees=cg_data.get("false_callees", []),
            verified_callers=cg_data.get("verified_callers", []),
            missing_callers=cg_data.get("missing_callers", []),
            false_callers=cg_data.get("false_callers", []),
        )

        # Parse corrections
        corrections = [
            CorrectionApplied(
                field=c.get("field", ""),
                action=c.get("action", "modified"),
                original_value=c.get("original_value"),
                corrected_value=c.get("corrected_value"),
                reason=c.get("reason", ""),
            )
            for c in data.get("corrections_applied", [])
        ]

        # Create metadata
        metadata = ValidatorMetadata(
            agent_id=self._config.agent_id,
            stream_id=self._config.stream_id,
            model=self._config.model_name,
            token_count=token_count,
        )

        return ValidationResult(
            component_id=component_id,
            validation_result=status,
            completeness=completeness,
            call_graph_accuracy=call_graph_accuracy,
            corrections_applied=corrections,
            metadata=metadata,
        )

    def _create_fallback_result(
        self,
        component_id: str,
        token_count: int,
    ) -> ValidationResult:
        """Create a minimal fallback result when parsing fails."""
        return ValidationResult(
            component_id=component_id,
            validation_result=ValidationStatus.WARNING,
            completeness=CompletenessCheck(),
            call_graph_accuracy=CallGraphAccuracy(),
            corrections_applied=[],
            metadata=ValidatorMetadata(
                agent_id=self._config.agent_id,
                stream_id=self._config.stream_id,
                model=self._config.model_name,
                token_count=token_count,
            ),
        )
