"""
Concrete Comparator Agent Implementation.

Implements the ComparatorAgent interface for comparing outputs from
documentation streams A and B, identifying discrepancies, and generating
Beads tickets for issues requiring human review.

Reference: Spec section 3.3
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime

from twinscribe.agents.comparator import (
    COMPARATOR_CONFIG,
    ComparatorAgent,
    ComparatorConfig,
    ComparatorInput,
)
from twinscribe.beads.client import BeadsClient, BeadsClientConfig, CreateIssueRequest
from twinscribe.beads.templates import (
    DiscrepancyTemplateData,
    TicketTemplateEngine,
)
from twinscribe.models.base import DiscrepancyType, ResolutionAction
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.comparison import (
    BeadsTicketRef,
    ComparatorMetadata,
    ComparisonResult,
    ComparisonSummary,
    ConvergenceStatus,
    Discrepancy,
)
from twinscribe.models.documentation import StreamOutput
from twinscribe.utils.llm_client import AsyncLLMClient, Message, get_comparator_client

logger = logging.getLogger(__name__)


class ConcreteComparatorAgent(ComparatorAgent):
    """Concrete implementation of the comparator agent.

    Compares outputs from Stream A and Stream B, identifies discrepancies,
    consults ground truth for call graph issues, and generates Beads tickets
    for issues requiring human review.

    Decision Logic:
        1. If stream_a == stream_b: ACCEPT (identical)
        2. If call_graph discrepancy: consult ground_truth, accept matching stream
        3. If documentation content discrepancy and confidence >= 0.7: accept better
        4. If confidence < 0.7: generate Beads ticket for human review

    Attributes:
        _llm_client: Async LLM client for comparison requests
        _beads_client: Beads client for ticket creation
        _template_engine: Template engine for ticket rendering
        _model_name: Model name used for comparisons
    """

    def __init__(
        self,
        config: ComparatorConfig | None = None,
        beads_config: BeadsClientConfig | None = None,
    ) -> None:
        """Initialize the comparator agent.

        Args:
            config: Comparator configuration (uses defaults if None)
            beads_config: Beads client configuration (uses defaults if None)
        """
        super().__init__(config or COMPARATOR_CONFIG)
        self._llm_client: AsyncLLMClient | None = None
        self._beads_client: BeadsClient | None = None
        self._beads_config = beads_config
        self._template_engine = TicketTemplateEngine()
        self._model_name: str = ""

    async def initialize(self) -> None:
        """Initialize the comparator agent.

        Sets up the LLM client and Beads client for ticket creation.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            logger.debug("Comparator agent already initialized")
            return

        try:
            # Initialize LLM client
            self._llm_client, self._model_name = get_comparator_client()
            await self._llm_client.__aenter__()

            # Initialize Beads client if ticket generation is enabled
            if self.comparator_config.generate_beads_tickets:
                self._beads_client = BeadsClient(self._beads_config)
                try:
                    await self._beads_client.initialize()
                except Exception as e:
                    logger.warning(
                        f"Beads client initialization failed: {e}. "
                        "Ticket generation will be disabled."
                    )
                    self._beads_client = None

            self._initialized = True
            self._metrics.started_at = datetime.utcnow()
            logger.info(f"Comparator agent initialized with model: {self._model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize comparator agent: {e}")
            raise RuntimeError(f"Comparator initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown the comparator agent.

        Closes LLM client and Beads client connections.
        """
        if not self._initialized:
            return

        self._metrics.completed_at = datetime.utcnow()

        # Close LLM client
        if self._llm_client:
            await self._llm_client.__aexit__(None, None, None)
            self._llm_client = None

        # Close Beads client
        if self._beads_client:
            await self._beads_client.close()
            self._beads_client = None

        self._initialized = False
        logger.info("Comparator agent shut down")

    async def compare(
        self,
        stream_a_output: StreamOutput,
        stream_b_output: StreamOutput,
        ground_truth_call_graph: CallGraph,
        iteration: int = 1,
    ) -> ComparisonResult:
        """Compare outputs from both streams.

        Convenience method that wraps process() for direct comparison calls.
        Constructs a ComparatorInput internally.

        Args:
            stream_a_output: Validated output from Stream A
            stream_b_output: Validated output from Stream B
            ground_truth_call_graph: Static analysis call graph (authoritative)
            iteration: Current iteration number (default 1)

        Returns:
            Comparison result with discrepancies and convergence status

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
        """
        input_data = ComparatorInput(
            stream_a_output=stream_a_output,
            stream_b_output=stream_b_output,
            ground_truth_call_graph=ground_truth_call_graph,
            iteration=iteration,
        )
        return await self.process(input_data)

    async def process(self, input_data: ComparatorInput) -> ComparisonResult:
        """Compare outputs from both streams.

        Main entry point for comparison. Compares each component from both
        streams, identifies discrepancies, resolves using ground truth where
        possible, and generates Beads tickets for unresolved issues.

        Args:
            input_data: Input containing both stream outputs and ground truth

        Returns:
            Comparison result with discrepancies and convergence status

        Raises:
            RuntimeError: If agent not initialized
            ValueError: If input is invalid
        """
        if not self._initialized:
            raise RuntimeError("Comparator agent not initialized. Call initialize() first.")

        start_time = time.time()

        # Validate input
        self._validate_input(input_data)

        # Generate comparison ID
        comparison_id = f"cmp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        # Collect all component IDs from both streams
        all_components = set(input_data.stream_a_output.outputs.keys()) | set(
            input_data.stream_b_output.outputs.keys()
        )

        # Track statistics
        identical_count = 0
        all_discrepancies: list[Discrepancy] = []
        identical_components: list[str] = []

        # Compare each component
        for component_id in all_components:
            stream_a_doc = self._get_component_doc(input_data.stream_a_output, component_id)
            stream_b_doc = self._get_component_doc(input_data.stream_b_output, component_id)

            # Check if identical first (fast path)
            if self._are_docs_identical(stream_a_doc, stream_b_doc):
                identical_count += 1
                identical_components.append(component_id)
                logger.debug(f"Component {component_id} is identical in both streams")
                continue

            # Detailed comparison for non-identical components
            discrepancies = await self.compare_component(
                component_id=component_id,
                stream_a_doc=stream_a_doc,
                stream_b_doc=stream_b_doc,
                ground_truth=input_data.ground_truth_call_graph,
            )

            # Set iteration for each discrepancy
            for disc in discrepancies:
                disc.iteration_found = input_data.iteration

            all_discrepancies.extend(discrepancies)

        # Resolve discrepancies and generate tickets
        resolved_by_gt = 0
        requires_human = 0

        for discrepancy in all_discrepancies:
            # Auto-resolve call graph discrepancies using ground truth
            if discrepancy.is_call_graph_related:
                self._resolve_with_ground_truth(discrepancy, input_data.ground_truth_call_graph)
                if discrepancy.is_resolved:
                    resolved_by_gt += 1
                    continue

            # Check if discrepancy was previously resolved
            if discrepancy.discrepancy_id in input_data.resolved_discrepancies:
                discrepancy.resolution = ResolutionAction.ACCEPT_STREAM_A  # Placeholder
                continue

            # Determine if human review is needed
            if (
                discrepancy.confidence < self.comparator_config.confidence_threshold
                and not discrepancy.is_resolved
            ):
                discrepancy.requires_beads = True
                requires_human += 1

                # Generate Beads ticket if enabled
                if self.comparator_config.generate_beads_tickets and self._beads_client:
                    ticket_data = await self.generate_beads_ticket(
                        discrepancy=discrepancy,
                        stream_a_model=self._get_stream_model(input_data.stream_a_output),
                        stream_b_model=self._get_stream_model(input_data.stream_b_output),
                        source_code="",  # Would need source code from somewhere
                    )
                    discrepancy.beads_ticket = BeadsTicketRef(
                        summary=ticket_data.get("title", ""),
                        description=ticket_data.get("description", ""),
                        priority=ticket_data.get("priority", "Medium"),
                        ticket_key=ticket_data.get("ticket_key"),
                    )

        # Calculate convergence status
        blocking_count = sum(1 for d in all_discrepancies if d.is_blocking)
        converged = blocking_count == 0 and len(all_discrepancies) == 0

        # Determine recommendation
        if converged:
            recommendation = "finalize"
        elif requires_human > 0:
            recommendation = "generate_beads_tickets"
        elif input_data.iteration >= 5:  # Max iterations check
            recommendation = "max_iterations_reached"
        else:
            recommendation = "continue"

        # Build result
        duration_ms = int((time.time() - start_time) * 1000)

        result = ComparisonResult(
            comparison_id=comparison_id,
            iteration=input_data.iteration,
            summary=ComparisonSummary(
                total_components=len(all_components),
                identical=identical_count,
                discrepancies=len(all_discrepancies),
                resolved_by_ground_truth=resolved_by_gt,
                requires_human_review=requires_human,
            ),
            discrepancies=all_discrepancies,
            convergence_status=ConvergenceStatus(
                converged=converged,
                blocking_discrepancies=blocking_count,
                recommendation=recommendation,
            ),
            metadata=ComparatorMetadata(
                agent_id=self.agent_id,
                model=self._model_name,
                comparison_duration_ms=duration_ms,
            ),
        )

        logger.info(
            f"Comparison completed: {identical_count}/{len(all_components)} identical, "
            f"{len(all_discrepancies)} discrepancies, converged={converged}"
        )

        return result

    async def compare_component(
        self,
        component_id: str,
        stream_a_doc: dict | None,
        stream_b_doc: dict | None,
        ground_truth: CallGraph,
    ) -> list[Discrepancy]:
        """Compare documentation for a single component.

        Performs detailed comparison between stream outputs for a single
        component, using LLM to identify semantic discrepancies.

        Args:
            component_id: Component being compared
            stream_a_doc: Documentation from Stream A (or None if missing)
            stream_b_doc: Documentation from Stream B (or None if missing)
            ground_truth: Static analysis call graph

        Returns:
            List of discrepancies found for this component
        """
        discrepancies: list[Discrepancy] = []

        # Handle missing documentation in one stream
        if stream_a_doc is None and stream_b_doc is None:
            logger.warning(f"No documentation for component {component_id} in either stream")
            return discrepancies

        if stream_a_doc is None:
            discrepancies.append(
                Discrepancy(
                    discrepancy_id=f"disc_{component_id}_missing_a",
                    component_id=component_id,
                    type=DiscrepancyType.DOCUMENTATION_CONTENT,
                    stream_a_value=None,
                    stream_b_value="present",
                    resolution=ResolutionAction.ACCEPT_STREAM_B,
                    confidence=0.9,
                )
            )
            return discrepancies

        if stream_b_doc is None:
            discrepancies.append(
                Discrepancy(
                    discrepancy_id=f"disc_{component_id}_missing_b",
                    component_id=component_id,
                    type=DiscrepancyType.DOCUMENTATION_CONTENT,
                    stream_a_value="present",
                    stream_b_value=None,
                    resolution=ResolutionAction.ACCEPT_STREAM_A,
                    confidence=0.9,
                )
            )
            return discrepancies

        # Get ground truth call graph info for this component
        gt_callees = ground_truth.get_callees(component_id)
        gt_callers = ground_truth.get_callers(component_id)

        # Compare call graphs first (can be resolved deterministically)
        call_graph_discrepancies = self._compare_call_graphs(
            component_id=component_id,
            stream_a_doc=stream_a_doc,
            stream_b_doc=stream_b_doc,
            gt_callees=gt_callees,
            gt_callers=gt_callers,
        )
        discrepancies.extend(call_graph_discrepancies)

        # Use LLM for semantic comparison of documentation content
        if self._llm_client:
            semantic_discrepancies = await self._compare_semantically(
                component_id=component_id,
                stream_a_doc=stream_a_doc,
                stream_b_doc=stream_b_doc,
                gt_callees=gt_callees,
                gt_callers=gt_callers,
            )
            discrepancies.extend(semantic_discrepancies)

        return discrepancies

    async def generate_beads_ticket(
        self,
        discrepancy: Discrepancy,
        stream_a_model: str,
        stream_b_model: str,
        source_code: str,
    ) -> dict:
        """Generate a Beads ticket for a discrepancy.

        Creates a structured ticket from the discrepancy data and submits
        it to Beads for human review.

        Args:
            discrepancy: The discrepancy requiring human review
            stream_a_model: Model name for Stream A
            stream_b_model: Model name for Stream B
            source_code: Relevant source code snippet

        Returns:
            Beads ticket data dict with ticket key if created
        """
        # Prepare template data
        template_data = DiscrepancyTemplateData(
            discrepancy_id=discrepancy.discrepancy_id,
            component_name=discrepancy.component_id,
            component_type="function",  # Would need to get from component metadata
            file_path="unknown",  # Would need to get from component metadata
            discrepancy_type=discrepancy.type.value,
            stream_a_value=str(discrepancy.stream_a_value) if discrepancy.stream_a_value else "N/A",
            stream_b_value=str(discrepancy.stream_b_value) if discrepancy.stream_b_value else "N/A",
            static_analysis_value=str(discrepancy.ground_truth)
            if discrepancy.ground_truth
            else None,
            context=f"Stream A Model: {stream_a_model}\nStream B Model: {stream_b_model}",
            iteration=discrepancy.iteration_found,
            priority=self._get_priority_from_discrepancy(discrepancy),
            labels=["ai-documentation", "twinscribe", f"discrepancy-{discrepancy.type.value}"],
        )

        # Render ticket content
        summary, description = self._template_engine.render_discrepancy(template_data)
        priority = self._template_engine.get_priority(template_data, "default_discrepancy")
        labels = self._template_engine.get_labels(template_data, "default_discrepancy")

        ticket_data = {
            "title": summary,
            "description": description,
            "priority": priority,
            "labels": labels,
            "ticket_key": None,
        }

        # Create ticket in Beads if client is available
        if self._beads_client:
            try:
                # Map priority to numeric value
                priority_map = {"Low": 2, "Medium": 1, "High": 0, "Critical": 0}
                priority_num = priority_map.get(priority, 1)

                request = CreateIssueRequest(
                    title=summary,
                    description=description,
                    priority=priority_num,
                    labels=labels,
                )

                issue = await self._beads_client.create_issue(request)
                ticket_data["ticket_key"] = issue.id
                logger.info(f"Created Beads ticket {issue.id} for {discrepancy.discrepancy_id}")

            except Exception as e:
                logger.error(f"Failed to create Beads ticket: {e}")

        return ticket_data

    def _validate_input(self, input_data: ComparatorInput) -> None:
        """Validate comparison input.

        Args:
            input_data: Input to validate

        Raises:
            ValueError: If input is invalid
        """
        if not input_data.stream_a_output:
            raise ValueError("stream_a_output is required")
        if not input_data.stream_b_output:
            raise ValueError("stream_b_output is required")
        if not input_data.ground_truth_call_graph:
            raise ValueError("ground_truth_call_graph is required")

    def _get_component_doc(self, stream_output: StreamOutput, component_id: str) -> dict | None:
        """Get component documentation as a dict.

        Args:
            stream_output: Stream output containing documentation
            component_id: Component ID to look up

        Returns:
            Documentation dict or None if not found
        """
        output = stream_output.get_output(component_id)
        if output is None:
            return None
        return output.model_dump()

    def _are_docs_identical(self, doc_a: dict | None, doc_b: dict | None) -> bool:
        """Check if two documentation dicts are identical.

        Performs structural comparison to quickly identify identical docs.

        Args:
            doc_a: First documentation dict
            doc_b: Second documentation dict

        Returns:
            True if docs are structurally identical
        """
        if doc_a is None and doc_b is None:
            return True
        if doc_a is None or doc_b is None:
            return False

        # Compare relevant fields (exclude metadata timestamps etc.)
        relevant_fields = ["documentation", "call_graph"]

        for field in relevant_fields:
            a_val = doc_a.get(field)
            b_val = doc_b.get(field)
            if a_val != b_val:
                return False

        return True

    def _compare_call_graphs(
        self,
        component_id: str,
        stream_a_doc: dict,
        stream_b_doc: dict,
        gt_callees: list,
        gt_callers: list,
    ) -> list[Discrepancy]:
        """Compare call graph information between streams.

        Args:
            component_id: Component being compared
            stream_a_doc: Stream A documentation
            stream_b_doc: Stream B documentation
            gt_callees: Ground truth callees
            gt_callers: Ground truth callers

        Returns:
            List of call graph discrepancies
        """
        discrepancies: list[Discrepancy] = []

        # Extract call graph from each stream
        a_call_graph = stream_a_doc.get("call_graph", {})
        b_call_graph = stream_b_doc.get("call_graph", {})

        a_callees = {c.get("component_id", "") for c in a_call_graph.get("callees", [])}
        b_callees = {c.get("component_id", "") for c in b_call_graph.get("callees", [])}
        gt_callee_ids = {e.callee for e in gt_callees}

        a_callers = {c.get("component_id", "") for c in a_call_graph.get("callers", [])}
        b_callers = {c.get("component_id", "") for c in b_call_graph.get("callers", [])}
        gt_caller_ids = {e.caller for e in gt_callers}

        # Check for discrepancies in callees
        if a_callees != b_callees:
            only_in_a = a_callees - b_callees
            only_in_b = b_callees - a_callees

            for callee in only_in_a | only_in_b:
                in_ground_truth = callee in gt_callee_ids
                a_has = callee in a_callees
                b_has = callee in b_callees

                # Determine resolution based on ground truth
                if in_ground_truth:
                    resolution, confidence = (
                        (ResolutionAction.ACCEPT_STREAM_A, 0.99)
                        if a_has
                        else (ResolutionAction.ACCEPT_STREAM_B, 0.99)
                    )
                else:
                    # Neither matches ground truth - edge should not exist
                    resolution = ResolutionAction.ACCEPT_GROUND_TRUTH
                    confidence = 0.99

                discrepancies.append(
                    Discrepancy(
                        discrepancy_id=f"disc_{component_id}_callee_{callee}_{uuid.uuid4().hex[:6]}",
                        component_id=component_id,
                        type=DiscrepancyType.CALL_GRAPH_EDGE,
                        stream_a_value=a_has,
                        stream_b_value=b_has,
                        ground_truth=in_ground_truth,
                        resolution=resolution,
                        confidence=confidence,
                    )
                )

        # Check for discrepancies in callers
        if a_callers != b_callers:
            only_in_a = a_callers - b_callers
            only_in_b = b_callers - a_callers

            for caller in only_in_a | only_in_b:
                in_ground_truth = caller in gt_caller_ids
                a_has = caller in a_callers
                b_has = caller in b_callers

                if in_ground_truth:
                    resolution, confidence = (
                        (ResolutionAction.ACCEPT_STREAM_A, 0.99)
                        if a_has
                        else (ResolutionAction.ACCEPT_STREAM_B, 0.99)
                    )
                else:
                    resolution = ResolutionAction.ACCEPT_GROUND_TRUTH
                    confidence = 0.99

                discrepancies.append(
                    Discrepancy(
                        discrepancy_id=f"disc_{component_id}_caller_{caller}_{uuid.uuid4().hex[:6]}",
                        component_id=component_id,
                        type=DiscrepancyType.CALL_GRAPH_EDGE,
                        stream_a_value=a_has,
                        stream_b_value=b_has,
                        ground_truth=in_ground_truth,
                        resolution=resolution,
                        confidence=confidence,
                    )
                )

        return discrepancies

    async def _compare_semantically(
        self,
        component_id: str,
        stream_a_doc: dict,
        stream_b_doc: dict,
        gt_callees: list,
        gt_callers: list,
    ) -> list[Discrepancy]:
        """Use LLM to compare documentation semantically.

        Args:
            component_id: Component being compared
            stream_a_doc: Stream A documentation
            stream_b_doc: Stream B documentation
            gt_callees: Ground truth callees
            gt_callers: Ground truth callers

        Returns:
            List of semantic discrepancies
        """
        if not self._llm_client:
            return []

        # Build comparison prompt
        prompt = self._build_comparison_prompt(
            component_id=component_id,
            stream_a_doc=stream_a_doc,
            stream_b_doc=stream_b_doc,
            gt_callees=gt_callees,
            gt_callers=gt_callers,
        )

        try:
            # Call LLM for comparison
            response = await self._llm_client.send_message(
                model=self._model_name,
                messages=[
                    Message(role="system", content=self.SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                json_mode=True,
                temperature=0.0,
            )

            # Record metrics
            self._metrics.record_request(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                latency_ms=float(response.latency_ms),
                cost=response.usage.cost_usd,
            )

            # Parse response
            result = json.loads(response.content)
            discrepancies = []

            for item in result.get("discrepancies", []):
                disc_type = self._map_discrepancy_type(item.get("type", "documentation_content"))
                resolution_str = item.get("resolution", "needs_human_review")
                resolution = self._map_resolution_action(resolution_str)

                discrepancy = Discrepancy(
                    discrepancy_id=item.get(
                        "discrepancy_id", f"disc_{component_id}_{uuid.uuid4().hex[:6]}"
                    ),
                    component_id=component_id,
                    type=disc_type,
                    stream_a_value=item.get("stream_a_value"),
                    stream_b_value=item.get("stream_b_value"),
                    ground_truth=item.get("ground_truth"),
                    resolution=resolution,
                    confidence=item.get("confidence", 0.5),
                    requires_beads=item.get("requires_beads", False),
                )

                # Add Beads ticket info if present in response
                if item.get("beads_ticket"):
                    ticket_info = item["beads_ticket"]
                    discrepancy.beads_ticket = BeadsTicketRef(
                        summary=ticket_info.get("summary", ""),
                        description=ticket_info.get("description", ""),
                        priority=ticket_info.get("priority", "Medium"),
                    )

                discrepancies.append(discrepancy)

            return discrepancies

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            self._metrics.record_error()
            return []
        except Exception as e:
            logger.error(f"Semantic comparison failed: {e}")
            self._metrics.record_error()
            return []

    def _resolve_with_ground_truth(self, discrepancy: Discrepancy, ground_truth: CallGraph) -> None:
        """Resolve a call graph discrepancy using ground truth.

        Modifies the discrepancy in place with resolution.

        Args:
            discrepancy: Discrepancy to resolve
            ground_truth: Ground truth call graph
        """
        if not discrepancy.is_call_graph_related:
            return

        gt_value = discrepancy.ground_truth
        a_value = discrepancy.stream_a_value
        b_value = discrepancy.stream_b_value

        resolution, confidence = self._determine_resolution(
            discrepancy_type=discrepancy.type.value,
            stream_a_value=a_value,
            stream_b_value=b_value,
            ground_truth=gt_value,
        )

        discrepancy.resolution = self._map_resolution_action(resolution)
        discrepancy.confidence = confidence

    def _get_stream_model(self, stream_output: StreamOutput) -> str:
        """Get model name from stream output.

        Args:
            stream_output: Stream output

        Returns:
            Model name string
        """
        # Get first output to find model name
        if stream_output.outputs:
            first_output = next(iter(stream_output.outputs.values()))
            return first_output.metadata.model
        return "unknown"

    def _get_priority_from_discrepancy(self, discrepancy: Discrepancy) -> str:
        """Determine ticket priority from discrepancy.

        Args:
            discrepancy: The discrepancy

        Returns:
            Priority string
        """
        # Call graph issues are higher priority (more impactful)
        if discrepancy.is_call_graph_related:
            return "High"

        # Low confidence issues need more attention
        if discrepancy.confidence < 0.5:
            return "High"

        return "Medium"

    def _map_discrepancy_type(self, type_str: str) -> DiscrepancyType:
        """Map string to DiscrepancyType enum.

        Args:
            type_str: Type string from LLM response

        Returns:
            DiscrepancyType enum value
        """
        type_map = {
            "call_graph_edge": DiscrepancyType.CALL_GRAPH_EDGE,
            "call_site_line": DiscrepancyType.CALL_SITE_LINE,
            "call_type_mismatch": DiscrepancyType.CALL_TYPE_MISMATCH,
            "documentation_content": DiscrepancyType.DOCUMENTATION_CONTENT,
            "parameter_description": DiscrepancyType.PARAMETER_DESCRIPTION,
            "return_description": DiscrepancyType.RETURN_DESCRIPTION,
            "exception_documentation": DiscrepancyType.EXCEPTION_DOCUMENTATION,
            "missing_parameter": DiscrepancyType.MISSING_PARAMETER,
            "missing_exception": DiscrepancyType.MISSING_EXCEPTION,
            "type_annotation": DiscrepancyType.TYPE_ANNOTATION,
        }
        return type_map.get(type_str.lower(), DiscrepancyType.DOCUMENTATION_CONTENT)

    def _map_resolution_action(self, resolution_str: str) -> ResolutionAction:
        """Map string to ResolutionAction enum.

        Args:
            resolution_str: Resolution string from LLM response

        Returns:
            ResolutionAction enum value
        """
        resolution_map = {
            "accept_stream_a": ResolutionAction.ACCEPT_STREAM_A,
            "accept_stream_b": ResolutionAction.ACCEPT_STREAM_B,
            "accept_ground_truth": ResolutionAction.ACCEPT_GROUND_TRUTH,
            "merge_both": ResolutionAction.MERGE_BOTH,
            "needs_human_review": ResolutionAction.NEEDS_HUMAN_REVIEW,
            "deferred": ResolutionAction.DEFERRED,
        }
        return resolution_map.get(resolution_str.lower(), ResolutionAction.NEEDS_HUMAN_REVIEW)


def create_comparator_agent(
    config: ComparatorConfig | None = None,
    beads_config: BeadsClientConfig | None = None,
) -> ConcreteComparatorAgent:
    """Factory function to create a comparator agent.

    Args:
        config: Comparator configuration (uses defaults if None)
        beads_config: Beads client configuration (uses defaults if None)

    Returns:
        Configured ConcreteComparatorAgent instance
    """
    return ConcreteComparatorAgent(config=config, beads_config=beads_config)
