# Comparator Agent and Arbitration Logic Design Document

## Version 1.0 | January 2026

---

## 1. Overview

This document specifies the architecture for the Comparator Agent (C) in the Dual-Stream Code Documentation System. The comparator is responsible for comparing outputs from both documentation streams, resolving discrepancies using ground truth and judgment, and generating Beads tickets when human review is required.

### Design Goals

1. **Accuracy First**: Call graph discrepancies resolved by static analysis ground truth
2. **Intelligent Arbitration**: Documentation content differences handled with judgment
3. **Transparent Decisions**: Every resolution includes rationale and confidence
4. **Human-in-Loop**: Low-confidence decisions escalate to Beads for review
5. **Convergence Tracking**: Clear criteria and progress toward convergence

---

## 2. Class Hierarchy

```
                    +------------------+
                    |   BaseAgent      |
                    |   <<abstract>>   |
                    +--------+---------+
                             |
              +--------------+
              |
    +---------+---------+
    | ComparatorAgent   |
    | (Claude Opus 4.5) |
    +---------+---------+
              |
              +---------------------------------------+
              |                                       |
    +---------+---------+                   +---------+---------+
    | DiscrepancyDetector|                  | ResolutionEngine  |
    | <<component>>      |                  | <<component>>     |
    +-------------------+                   +-------------------+
              |                                       |
    +---------+---------+                   +---------+---------+
    | StructuralComparator|                 | BeadsTicketGenerator|
    | SemanticComparator  |                 | ConvergenceTracker  |
    +-------------------+                   +-------------------+
```

---

## 3. Core Data Models

```python
# twinscribe/agents/comparator/models.py

from datetime import datetime
from typing import Optional, Any
from enum import Enum

from pydantic import BaseModel, Field


class DiscrepancyType(str, Enum):
    """Types of discrepancies that can occur between streams."""
    # Call graph discrepancies
    CALL_GRAPH_EDGE = "call_graph_edge"
    MISSING_CALLEE = "missing_callee"
    FALSE_CALLEE = "false_callee"
    MISSING_CALLER = "missing_caller"
    FALSE_CALLER = "false_caller"
    CALL_TYPE_MISMATCH = "call_type_mismatch"
    LINE_NUMBER_MISMATCH = "line_number_mismatch"

    # Documentation content discrepancies
    DOCUMENTATION_CONTENT = "documentation_content"
    SUMMARY_MISMATCH = "summary_mismatch"
    DESCRIPTION_MISMATCH = "description_mismatch"
    PARAMETER_MISMATCH = "parameter_mismatch"
    RETURN_TYPE_MISMATCH = "return_type_mismatch"
    EXCEPTION_MISMATCH = "exception_mismatch"

    # Structural discrepancies
    MISSING_COMPONENT = "missing_component"
    EXTRA_COMPONENT = "extra_component"


class ResolutionType(str, Enum):
    """How a discrepancy was resolved."""
    ACCEPT_STREAM_A = "accept_stream_a"
    ACCEPT_STREAM_B = "accept_stream_b"
    ACCEPT_GROUND_TRUTH = "accept_ground_truth"
    MERGE_BOTH = "merge_both"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    IDENTICAL = "identical"


class DiscrepancySeverity(str, Enum):
    """Severity of a discrepancy."""
    CRITICAL = "critical"  # Blocking, must be resolved
    HIGH = "high"          # Important, should be resolved
    MEDIUM = "medium"      # Noticeable, nice to resolve
    LOW = "low"            # Minor, can be ignored


class BlockingDiscrepancyType(str, Enum):
    """Types that prevent convergence."""
    MISSING_CRITICAL_CALL = "missing_critical_call"
    FALSE_CRITICAL_CALL = "false_critical_call"
    MISSING_PUBLIC_API_DOC = "missing_public_api_doc"
    SECURITY_RELEVANT_GAP = "security_relevant_gap"


class BeadsTicketInfo(BaseModel):
    """Information for creating a Beads ticket."""
    summary: str
    description: str
    priority: str = "Medium"
    labels: list[str] = Field(default_factory=list)
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class Discrepancy(BaseModel):
    """A single discrepancy between streams."""
    discrepancy_id: str
    component_id: str
    type: DiscrepancyType
    severity: DiscrepancySeverity
    stream_a_value: Any
    stream_b_value: Any
    ground_truth: Optional[Any] = None
    resolution: ResolutionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    requires_beads: bool = False
    beads_ticket: Optional[BeadsTicketInfo] = None
    rationale: str = ""


class ComparisonSummary(BaseModel):
    """Summary statistics for a comparison."""
    total_components: int
    identical: int
    discrepancies: int
    resolved_by_ground_truth: int
    resolved_by_judgment: int
    requires_human_review: int


class ConvergenceStatus(BaseModel):
    """Status of convergence between streams."""
    converged: bool
    blocking_discrepancies: int
    call_graph_match_rate: float
    documentation_similarity: float
    open_discrepancies: int
    recommendation: str  # "continue", "generate_beads_tickets", "force_converge"


class ComparatorInput(BaseModel):
    """Input to the comparator agent."""
    stream_a_output: dict[str, Any]  # component_id -> DocumenterOutput
    stream_b_output: dict[str, Any]
    static_call_graph: dict[str, Any]  # Ground truth
    iteration: int
    previous_comparison: Optional["ComparisonResult"] = None

    def to_prompt_context(self) -> str:
        """Convert to prompt context for LLM."""
        return f"""
Iteration: {self.iteration}
Total Components Stream A: {len(self.stream_a_output)}
Total Components Stream B: {len(self.stream_b_output)}
Static Analysis Edges: {len(self.static_call_graph.get('edges', []))}

Previous iteration discrepancies: {
    self.previous_comparison.summary.discrepancies if self.previous_comparison else 'N/A'
}
"""

    def get_component_id(self) -> str:
        return "comparison"


class ComparisonResult(BaseModel):
    """Complete result of comparing both streams."""
    comparison_id: str
    iteration: int
    summary: ComparisonSummary
    discrepancies: list[Discrepancy]
    convergence_status: ConvergenceStatus
    corrections_for_stream_a: dict[str, Any] = Field(default_factory=dict)
    corrections_for_stream_b: dict[str, Any] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()

    def get_component_id(self) -> str:
        return self.comparison_id

    def get_confidence(self) -> float:
        if not self.discrepancies:
            return 1.0
        return sum(d.confidence for d in self.discrepancies) / len(self.discrepancies)
```

---

## 4. Convergence Criteria

```python
# twinscribe/agents/comparator/convergence.py

from dataclasses import dataclass
from typing import Optional

from twinscribe.agents.comparator.models import (
    ComparisonResult,
    ConvergenceStatus,
    BlockingDiscrepancyType,
    DiscrepancyType,
)


@dataclass
class ConvergenceCriteria:
    """
    Criteria for determining when streams have converged.

    Hard Thresholds (must all be met for convergence):
    - max_iterations: Absolute maximum iterations allowed
    - call_graph_match_rate: Minimum % of call graph edges that must match
    - documentation_similarity: Minimum semantic similarity score
    - max_open_discrepancies: Maximum unresolved non-blocking issues

    Blocking Conditions (prevent convergence regardless of scores):
    - missing_critical_call: Call exists in code but undocumented
    - false_critical_call: Documented call doesn't exist
    - missing_public_api_doc: Public method undocumented
    - security_relevant_gap: Security-sensitive code undocumented
    """

    # Hard thresholds
    max_iterations: int = 5
    call_graph_match_rate: float = 0.98  # 98% edges must match
    documentation_similarity: float = 0.95  # 95% semantic similarity
    max_open_discrepancies: int = 2  # Max unresolved non-blocking

    # Blocking discrepancy types
    blocking_types: tuple[BlockingDiscrepancyType, ...] = (
        BlockingDiscrepancyType.MISSING_CRITICAL_CALL,
        BlockingDiscrepancyType.FALSE_CRITICAL_CALL,
        BlockingDiscrepancyType.MISSING_PUBLIC_API_DOC,
        BlockingDiscrepancyType.SECURITY_RELEVANT_GAP,
    )

    def check_convergence(
        self,
        comparison: ComparisonResult,
        iteration: int,
    ) -> ConvergenceStatus:
        """
        Check if convergence criteria are met.

        Returns:
            ConvergenceStatus with convergence state and recommendation
        """
        # Count blocking discrepancies
        blocking_count = self._count_blocking_discrepancies(comparison)

        # Calculate match rates
        call_graph_rate = self._calculate_call_graph_match_rate(comparison)
        doc_similarity = self._calculate_documentation_similarity(comparison)

        # Count open (unresolved) discrepancies
        open_count = sum(
            1 for d in comparison.discrepancies
            if d.resolution == ResolutionType.NEEDS_HUMAN_REVIEW
        )

        # Determine convergence
        converged = (
            blocking_count == 0 and
            call_graph_rate >= self.call_graph_match_rate and
            doc_similarity >= self.documentation_similarity and
            open_count <= self.max_open_discrepancies
        )

        # Determine recommendation
        if converged:
            recommendation = "converged"
        elif iteration >= self.max_iterations:
            recommendation = "force_converge"
        elif blocking_count > 0 or open_count > self.max_open_discrepancies:
            recommendation = "generate_beads_tickets"
        else:
            recommendation = "continue"

        return ConvergenceStatus(
            converged=converged,
            blocking_discrepancies=blocking_count,
            call_graph_match_rate=call_graph_rate,
            documentation_similarity=doc_similarity,
            open_discrepancies=open_count,
            recommendation=recommendation,
        )

    def _count_blocking_discrepancies(
        self,
        comparison: ComparisonResult
    ) -> int:
        """Count discrepancies that block convergence."""
        count = 0
        for disc in comparison.discrepancies:
            if disc.resolution == ResolutionType.NEEDS_HUMAN_REVIEW:
                # Check if this is a blocking type
                if self._is_blocking_discrepancy(disc):
                    count += 1
        return count

    def _is_blocking_discrepancy(self, disc: Discrepancy) -> bool:
        """Determine if a discrepancy is blocking."""
        # Call graph discrepancies involving critical calls
        if disc.type in (
            DiscrepancyType.MISSING_CALLEE,
            DiscrepancyType.FALSE_CALLEE,
        ):
            return self._is_critical_call(disc.stream_a_value or disc.stream_b_value)

        # Missing documentation for public API
        if disc.type == DiscrepancyType.MISSING_COMPONENT:
            return self._is_public_api(disc.component_id)

        return False

    def _is_critical_call(self, call_info: Any) -> bool:
        """Determine if a call is critical (affects program flow significantly)."""
        # Implementation would check if the call is to:
        # - Error handling
        # - Security functions
        # - Data validation
        # - Core business logic
        return False  # Placeholder

    def _is_public_api(self, component_id: str) -> bool:
        """Determine if a component is part of the public API."""
        # Check naming conventions, decorators, etc.
        return not component_id.startswith("_")

    def _calculate_call_graph_match_rate(
        self,
        comparison: ComparisonResult
    ) -> float:
        """Calculate the percentage of matching call graph edges."""
        total = comparison.summary.total_components
        if total == 0:
            return 1.0

        call_graph_discrepancies = sum(
            1 for d in comparison.discrepancies
            if d.type in (
                DiscrepancyType.CALL_GRAPH_EDGE,
                DiscrepancyType.MISSING_CALLEE,
                DiscrepancyType.FALSE_CALLEE,
                DiscrepancyType.MISSING_CALLER,
                DiscrepancyType.FALSE_CALLER,
            )
        )

        # Estimate total edges (rough approximation)
        estimated_edges = total * 3  # Assume average 3 call relationships per component
        if estimated_edges == 0:
            return 1.0

        return 1.0 - (call_graph_discrepancies / estimated_edges)

    def _calculate_documentation_similarity(
        self,
        comparison: ComparisonResult
    ) -> float:
        """Calculate semantic similarity of documentation content."""
        total = comparison.summary.total_components
        if total == 0:
            return 1.0

        content_discrepancies = sum(
            1 for d in comparison.discrepancies
            if d.type in (
                DiscrepancyType.DOCUMENTATION_CONTENT,
                DiscrepancyType.SUMMARY_MISMATCH,
                DiscrepancyType.DESCRIPTION_MISMATCH,
            )
        )

        return 1.0 - (content_discrepancies / total)
```

---

## 5. Comparator Agent Implementation

```python
# twinscribe/agents/comparator/agent.py

import json
import re
import uuid
from datetime import datetime
from typing import Optional, Any

from twinscribe.agents.base import BaseAgent, AgentMetadata
from twinscribe.agents.comparator.models import (
    ComparatorInput,
    ComparisonResult,
    ComparisonSummary,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ResolutionType,
    BeadsTicketInfo,
)
from twinscribe.agents.comparator.convergence import ConvergenceCriteria
from twinscribe.agents.comparator.detector import DiscrepancyDetector
from twinscribe.agents.comparator.resolver import ResolutionEngine
from twinscribe.agents.prompts import COMPARATOR_SYSTEM_PROMPT
from twinscribe.analysis.static_oracle import StaticAnalysisOracle
from twinscribe.config.models import ModelConfig, ModelTier


class ComparatorAgent(BaseAgent[ComparatorInput, ComparisonResult]):
    """
    Agent responsible for comparing outputs from both documentation streams
    and resolving discrepancies.

    Model Tier: Arbitration (Claude Opus 4.5)

    Responsibilities:
    - Compare outputs component-by-component
    - Identify all structural and semantic discrepancies
    - Consult static analysis for call graph discrepancies
    - Use judgment for documentation content differences
    - Generate Beads tickets for low-confidence decisions
    - Track convergence progress

    Decision Hierarchy:
    1. Static analysis is AUTHORITATIVE for call graph accuracy
    2. For semantic/content differences, prefer completeness and accuracy
    3. When uncertain (confidence < 0.7), generate Beads ticket
    4. Never guess - escalate unclear cases
    """

    CONFIDENCE_THRESHOLD = 0.7  # Below this, generate Beads ticket

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        static_oracle: StaticAnalysisOracle,
        convergence_criteria: Optional[ConvergenceCriteria] = None,
        **kwargs
    ):
        model_config = ModelConfig(
            tier=ModelTier.ARBITRATION,
            provider="anthropic",
            model_name="claude-opus-4-5-20251101",
            cost_per_million=15.0,
        )
        super().__init__(
            agent_id=agent_id,
            model_config=model_config,
            llm_client=llm_client,
            system_prompt=COMPARATOR_SYSTEM_PROMPT,
            **kwargs,
        )
        self.static_oracle = static_oracle
        self.convergence_criteria = convergence_criteria or ConvergenceCriteria()
        self.detector = DiscrepancyDetector(static_oracle)
        self.resolver = ResolutionEngine(static_oracle, self.CONFIDENCE_THRESHOLD)

    @property
    def agent_type(self) -> str:
        return "comparator"

    async def compare(
        self,
        stream_a_output: dict[str, Any],
        stream_b_output: dict[str, Any],
        iteration: int = 1,
        previous_comparison: Optional[ComparisonResult] = None,
    ) -> ComparisonResult:
        """
        Compare outputs from both streams and produce comparison result.

        This is the main entry point for comparison logic.

        Args:
            stream_a_output: Documentation outputs from Stream A
            stream_b_output: Documentation outputs from Stream B
            iteration: Current iteration number
            previous_comparison: Results from previous iteration (if any)

        Returns:
            ComparisonResult with discrepancies and resolutions
        """
        comparison_id = f"cmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{iteration:03d}"

        # Step 1: Detect all discrepancies
        discrepancies = await self._detect_discrepancies(
            stream_a_output,
            stream_b_output,
        )

        # Step 2: Resolve discrepancies
        resolved_discrepancies = await self._resolve_discrepancies(
            discrepancies,
            stream_a_output,
            stream_b_output,
        )

        # Step 3: Build summary
        summary = self._build_summary(
            stream_a_output,
            stream_b_output,
            resolved_discrepancies,
        )

        # Step 4: Check convergence
        preliminary_result = ComparisonResult(
            comparison_id=comparison_id,
            iteration=iteration,
            summary=summary,
            discrepancies=resolved_discrepancies,
            convergence_status=ConvergenceStatus(
                converged=False,
                blocking_discrepancies=0,
                call_graph_match_rate=0.0,
                documentation_similarity=0.0,
                open_discrepancies=0,
                recommendation="pending",
            ),
            metadata={
                "agent_id": self.agent_id,
                "model": self.model_config.model_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

        convergence_status = self.convergence_criteria.check_convergence(
            preliminary_result,
            iteration,
        )

        # Step 5: Build corrections
        corrections_a, corrections_b = self._build_corrections(resolved_discrepancies)

        return ComparisonResult(
            comparison_id=comparison_id,
            iteration=iteration,
            summary=summary,
            discrepancies=resolved_discrepancies,
            convergence_status=convergence_status,
            corrections_for_stream_a=corrections_a,
            corrections_for_stream_b=corrections_b,
            metadata={
                "agent_id": self.agent_id,
                "model": self.model_config.model_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _detect_discrepancies(
        self,
        stream_a_output: dict[str, Any],
        stream_b_output: dict[str, Any],
    ) -> list[Discrepancy]:
        """Detect all discrepancies between streams."""
        discrepancies = []

        # Get all component IDs from both streams
        all_components = set(stream_a_output.keys()) | set(stream_b_output.keys())

        for component_id in all_components:
            a_doc = stream_a_output.get(component_id)
            b_doc = stream_b_output.get(component_id)

            # Check for missing components
            if a_doc is None:
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.MISSING_COMPONENT,
                    severity=DiscrepancySeverity.HIGH,
                    stream_a_value=None,
                    stream_b_value=b_doc,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=0.5,
                    requires_beads=True,
                    rationale="Component missing from Stream A",
                ))
                continue

            if b_doc is None:
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.MISSING_COMPONENT,
                    severity=DiscrepancySeverity.HIGH,
                    stream_a_value=a_doc,
                    stream_b_value=None,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=0.5,
                    requires_beads=True,
                    rationale="Component missing from Stream B",
                ))
                continue

            # Detect call graph discrepancies
            call_graph_discs = self.detector.detect_call_graph_discrepancies(
                component_id, a_doc, b_doc
            )
            discrepancies.extend(call_graph_discs)

            # Detect documentation content discrepancies
            content_discs = self.detector.detect_content_discrepancies(
                component_id, a_doc, b_doc
            )
            discrepancies.extend(content_discs)

        return discrepancies

    async def _resolve_discrepancies(
        self,
        discrepancies: list[Discrepancy],
        stream_a_output: dict[str, Any],
        stream_b_output: dict[str, Any],
    ) -> list[Discrepancy]:
        """Resolve each discrepancy using appropriate strategy."""
        resolved = []

        for disc in discrepancies:
            if disc.type in (
                DiscrepancyType.CALL_GRAPH_EDGE,
                DiscrepancyType.MISSING_CALLEE,
                DiscrepancyType.FALSE_CALLEE,
                DiscrepancyType.MISSING_CALLER,
                DiscrepancyType.FALSE_CALLER,
            ):
                # Use ground truth for call graph
                resolved_disc = self.resolver.resolve_by_ground_truth(disc)
            else:
                # Use LLM judgment for content discrepancies
                resolved_disc = await self._resolve_with_llm(
                    disc,
                    stream_a_output.get(disc.component_id),
                    stream_b_output.get(disc.component_id),
                )

            # Check if Beads ticket needed
            if resolved_disc.confidence < self.CONFIDENCE_THRESHOLD:
                resolved_disc = self._prepare_beads_ticket(resolved_disc)

            resolved.append(resolved_disc)

        return resolved

    async def _resolve_with_llm(
        self,
        disc: Discrepancy,
        a_doc: Optional[dict],
        b_doc: Optional[dict],
    ) -> Discrepancy:
        """Use LLM to resolve content discrepancy."""
        input_data = ComparatorInput(
            stream_a_output={disc.component_id: a_doc} if a_doc else {},
            stream_b_output={disc.component_id: b_doc} if b_doc else {},
            static_call_graph={},
            iteration=1,
        )

        # Build focused prompt for this specific discrepancy
        prompt = self._build_resolution_prompt(disc, a_doc, b_doc)

        try:
            response = await self.llm_client.complete(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                model=self.model_config.model_name,
                temperature=0.2,  # Low temperature for consistent judgment
                max_tokens=2048,
            )

            resolution_data = self._parse_resolution_response(response.content)

            return Discrepancy(
                discrepancy_id=disc.discrepancy_id,
                component_id=disc.component_id,
                type=disc.type,
                severity=disc.severity,
                stream_a_value=disc.stream_a_value,
                stream_b_value=disc.stream_b_value,
                ground_truth=disc.ground_truth,
                resolution=ResolutionType(resolution_data.get("resolution", "needs_human_review")),
                confidence=resolution_data.get("confidence", 0.5),
                requires_beads=resolution_data.get("confidence", 0.5) < self.CONFIDENCE_THRESHOLD,
                rationale=resolution_data.get("rationale", ""),
            )

        except Exception as e:
            # On error, escalate to human review
            return Discrepancy(
                discrepancy_id=disc.discrepancy_id,
                component_id=disc.component_id,
                type=disc.type,
                severity=disc.severity,
                stream_a_value=disc.stream_a_value,
                stream_b_value=disc.stream_b_value,
                ground_truth=disc.ground_truth,
                resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                confidence=0.0,
                requires_beads=True,
                rationale=f"Resolution failed: {e}",
            )

    def _build_resolution_prompt(
        self,
        disc: Discrepancy,
        a_doc: Optional[dict],
        b_doc: Optional[dict],
    ) -> str:
        """Build prompt for resolving a specific discrepancy."""
        return f"""
Please resolve the following discrepancy:

Component: {disc.component_id}
Discrepancy Type: {disc.type.value}
Severity: {disc.severity.value}

Stream A Value:
{json.dumps(disc.stream_a_value, indent=2) if disc.stream_a_value else "None"}

Stream B Value:
{json.dumps(disc.stream_b_value, indent=2) if disc.stream_b_value else "None"}

Full Stream A Documentation:
{json.dumps(a_doc, indent=2) if a_doc else "Not available"}

Full Stream B Documentation:
{json.dumps(b_doc, indent=2) if b_doc else "Not available"}

Please analyze both values and determine which is more accurate/complete.
Respond with JSON:
{{
  "resolution": "accept_stream_a" | "accept_stream_b" | "merge_both" | "needs_human_review",
  "confidence": 0.0-1.0,
  "rationale": "explanation of your decision",
  "merged_value": {{...}} // only if resolution is "merge_both"
}}
"""

    def _parse_resolution_response(self, response: str) -> dict:
        """Parse LLM resolution response."""
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "resolution": "needs_human_review",
                "confidence": 0.0,
                "rationale": "Failed to parse resolution response",
            }

    def _prepare_beads_ticket(self, disc: Discrepancy) -> Discrepancy:
        """Prepare Beads ticket information for a discrepancy."""
        ticket_info = BeadsTicketInfo(
            summary=f"[AI-DOC] {disc.type.value}: {disc.component_id}",
            description=self._build_ticket_description(disc),
            priority=self._determine_ticket_priority(disc.severity),
            labels=["ai-documentation", disc.type.value],
            custom_fields={
                "cf_component_id": disc.component_id,
                "cf_stream_a_confidence": 0.0,  # Would come from metadata
                "cf_stream_b_confidence": 0.0,
                "cf_auto_resolvable": False,
            },
        )

        return Discrepancy(
            discrepancy_id=disc.discrepancy_id,
            component_id=disc.component_id,
            type=disc.type,
            severity=disc.severity,
            stream_a_value=disc.stream_a_value,
            stream_b_value=disc.stream_b_value,
            ground_truth=disc.ground_truth,
            resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
            confidence=disc.confidence,
            requires_beads=True,
            beads_ticket=ticket_info,
            rationale=disc.rationale,
        )

    def _build_ticket_description(self, disc: Discrepancy) -> str:
        """Build Beads ticket description from discrepancy."""
        return f"""## Discrepancy Summary

**Component:** `{disc.component_id}`
**Type:** {disc.type.value}
**Severity:** {disc.severity.value}
**Confidence:** {disc.confidence:.2f}

## Stream Comparison

### Stream A Value
```json
{json.dumps(disc.stream_a_value, indent=2) if disc.stream_a_value else "null"}
```

### Stream B Value
```json
{json.dumps(disc.stream_b_value, indent=2) if disc.stream_b_value else "null"}
```

### Ground Truth (if available)
```json
{json.dumps(disc.ground_truth, indent=2) if disc.ground_truth else "Not available for this discrepancy type"}
```

## Analysis
{disc.rationale}

## Requested Action

Please review and indicate which interpretation is correct:
- [ ] Stream A is correct
- [ ] Stream B is correct
- [ ] Both are partially correct (provide merged version)
- [ ] Neither is correct (provide correct version)
"""

    def _determine_ticket_priority(self, severity: DiscrepancySeverity) -> str:
        """Map discrepancy severity to Beads priority."""
        mapping = {
            DiscrepancySeverity.CRITICAL: "Highest",
            DiscrepancySeverity.HIGH: "High",
            DiscrepancySeverity.MEDIUM: "Medium",
            DiscrepancySeverity.LOW: "Low",
        }
        return mapping.get(severity, "Medium")

    def _build_summary(
        self,
        stream_a_output: dict[str, Any],
        stream_b_output: dict[str, Any],
        discrepancies: list[Discrepancy],
    ) -> ComparisonSummary:
        """Build comparison summary statistics."""
        total_components = len(set(stream_a_output.keys()) | set(stream_b_output.keys()))

        identical = total_components - len(discrepancies)

        resolved_by_ground_truth = sum(
            1 for d in discrepancies
            if d.resolution == ResolutionType.ACCEPT_GROUND_TRUTH
        )

        resolved_by_judgment = sum(
            1 for d in discrepancies
            if d.resolution in (
                ResolutionType.ACCEPT_STREAM_A,
                ResolutionType.ACCEPT_STREAM_B,
                ResolutionType.MERGE_BOTH,
            )
        )

        requires_human_review = sum(
            1 for d in discrepancies
            if d.resolution == ResolutionType.NEEDS_HUMAN_REVIEW
        )

        return ComparisonSummary(
            total_components=total_components,
            identical=identical,
            discrepancies=len(discrepancies),
            resolved_by_ground_truth=resolved_by_ground_truth,
            resolved_by_judgment=resolved_by_judgment,
            requires_human_review=requires_human_review,
        )

    def _build_corrections(
        self,
        discrepancies: list[Discrepancy]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build correction dictionaries for each stream."""
        corrections_a: dict[str, Any] = {}
        corrections_b: dict[str, Any] = {}

        for disc in discrepancies:
            if disc.resolution == ResolutionType.ACCEPT_STREAM_A:
                # Stream B needs correction
                if disc.component_id not in corrections_b:
                    corrections_b[disc.component_id] = {}
                corrections_b[disc.component_id][disc.type.value] = disc.stream_a_value

            elif disc.resolution == ResolutionType.ACCEPT_STREAM_B:
                # Stream A needs correction
                if disc.component_id not in corrections_a:
                    corrections_a[disc.component_id] = {}
                corrections_a[disc.component_id][disc.type.value] = disc.stream_b_value

            elif disc.resolution == ResolutionType.ACCEPT_GROUND_TRUTH:
                # Both streams need ground truth
                if disc.ground_truth is not None:
                    if disc.stream_a_value != disc.ground_truth:
                        if disc.component_id not in corrections_a:
                            corrections_a[disc.component_id] = {}
                        corrections_a[disc.component_id][disc.type.value] = disc.ground_truth

                    if disc.stream_b_value != disc.ground_truth:
                        if disc.component_id not in corrections_b:
                            corrections_b[disc.component_id] = {}
                        corrections_b[disc.component_id][disc.type.value] = disc.ground_truth

        return corrections_a, corrections_b

    # Required abstract method implementations
    async def _build_prompt(self, input_data: ComparatorInput) -> str:
        """Build the comparison prompt."""
        return f"""
Compare the following documentation outputs from two independent streams.

{input_data.to_prompt_context()}

Identify all discrepancies and resolve them according to the decision hierarchy:
1. Call graph discrepancies: Use static analysis ground truth
2. Content discrepancies: Use judgment based on completeness and accuracy
3. Low confidence (<0.7): Mark for human review

Respond with JSON containing discrepancies and resolutions.
"""

    def _parse_response(
        self,
        response: str,
        input_data: ComparatorInput
    ) -> ComparisonResult:
        """Parse comparison response (not typically used - compare() is main entry)."""
        # This is implemented for interface compliance but compare() is the main API
        raise NotImplementedError("Use compare() method instead")

    def _validate_output(self, output: ComparisonResult) -> list[str]:
        """Validate comparison result."""
        errors = []

        if output.summary.total_components < 0:
            errors.append("Invalid total_components count")

        for disc in output.discrepancies:
            if not disc.discrepancy_id:
                errors.append("Discrepancy missing ID")
            if not disc.rationale and disc.resolution != ResolutionType.IDENTICAL:
                errors.append(f"Missing rationale for {disc.discrepancy_id}")

        return errors
```

---

## 6. Discrepancy Detector Component

```python
# twinscribe/agents/comparator/detector.py

import uuid
from typing import Any, Optional

from twinscribe.agents.comparator.models import (
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    ResolutionType,
)
from twinscribe.analysis.static_oracle import StaticAnalysisOracle


class DiscrepancyDetector:
    """
    Component responsible for detecting discrepancies between stream outputs.

    Types of detection:
    - Structural: Missing/extra components, schema mismatches
    - Call Graph: Edge differences against ground truth
    - Content: Semantic differences in documentation text
    """

    def __init__(self, static_oracle: StaticAnalysisOracle):
        self.static_oracle = static_oracle

    def detect_call_graph_discrepancies(
        self,
        component_id: str,
        a_doc: dict,
        b_doc: dict,
    ) -> list[Discrepancy]:
        """Detect call graph discrepancies between streams."""
        discrepancies = []

        a_callees = set(
            c.get("component_id")
            for c in a_doc.get("call_graph", {}).get("callees", [])
        )
        b_callees = set(
            c.get("component_id")
            for c in b_doc.get("call_graph", {}).get("callees", [])
        )

        # Get ground truth
        ground_truth_callees = set(
            e.callee for e in self.static_oracle.get_callees(component_id)
        )

        # Callees in A but not in B
        for callee in a_callees - b_callees:
            in_ground_truth = callee in ground_truth_callees
            discrepancies.append(Discrepancy(
                discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                component_id=component_id,
                type=DiscrepancyType.CALL_GRAPH_EDGE,
                severity=DiscrepancySeverity.HIGH if in_ground_truth else DiscrepancySeverity.MEDIUM,
                stream_a_value={"callee": callee},
                stream_b_value=None,
                ground_truth={"exists": in_ground_truth},
                resolution=ResolutionType.NEEDS_HUMAN_REVIEW,  # Will be resolved later
                confidence=0.0,
                rationale=f"Callee {callee} in Stream A only",
            ))

        # Callees in B but not in A
        for callee in b_callees - a_callees:
            in_ground_truth = callee in ground_truth_callees
            discrepancies.append(Discrepancy(
                discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                component_id=component_id,
                type=DiscrepancyType.CALL_GRAPH_EDGE,
                severity=DiscrepancySeverity.HIGH if in_ground_truth else DiscrepancySeverity.MEDIUM,
                stream_a_value=None,
                stream_b_value={"callee": callee},
                ground_truth={"exists": in_ground_truth},
                resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                confidence=0.0,
                rationale=f"Callee {callee} in Stream B only",
            ))

        # Check for false callees (in both streams but not in ground truth)
        common_callees = a_callees & b_callees
        for callee in common_callees:
            if callee not in ground_truth_callees:
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.FALSE_CALLEE,
                    severity=DiscrepancySeverity.HIGH,
                    stream_a_value={"callee": callee},
                    stream_b_value={"callee": callee},
                    ground_truth={"exists": False},
                    resolution=ResolutionType.ACCEPT_GROUND_TRUTH,
                    confidence=0.99,  # High confidence - ground truth says it doesn't exist
                    rationale=f"Both streams claim callee {callee} but static analysis disagrees",
                ))

        # Check for missing callees (in ground truth but not in either stream)
        missing = ground_truth_callees - (a_callees | b_callees)
        for callee in missing:
            discrepancies.append(Discrepancy(
                discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                component_id=component_id,
                type=DiscrepancyType.MISSING_CALLEE,
                severity=DiscrepancySeverity.CRITICAL,  # Both streams missed it
                stream_a_value=None,
                stream_b_value=None,
                ground_truth={"callee": callee, "exists": True},
                resolution=ResolutionType.ACCEPT_GROUND_TRUTH,
                confidence=0.99,
                rationale=f"Neither stream documented callee {callee} found in static analysis",
            ))

        return discrepancies

    def detect_content_discrepancies(
        self,
        component_id: str,
        a_doc: dict,
        b_doc: dict,
    ) -> list[Discrepancy]:
        """Detect documentation content discrepancies."""
        discrepancies = []

        a_content = a_doc.get("documentation", {})
        b_content = b_doc.get("documentation", {})

        # Check summary
        a_summary = a_content.get("summary", "")
        b_summary = b_content.get("summary", "")
        if a_summary != b_summary:
            similarity = self._calculate_similarity(a_summary, b_summary)
            if similarity < 0.9:  # Significant difference
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.SUMMARY_MISMATCH,
                    severity=DiscrepancySeverity.MEDIUM,
                    stream_a_value=a_summary,
                    stream_b_value=b_summary,
                    ground_truth=None,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=similarity,
                    rationale=f"Summary differs (similarity: {similarity:.2f})",
                ))

        # Check description
        a_desc = a_content.get("description", "")
        b_desc = b_content.get("description", "")
        if a_desc != b_desc:
            similarity = self._calculate_similarity(a_desc, b_desc)
            if similarity < 0.85:  # More tolerance for descriptions
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.DESCRIPTION_MISMATCH,
                    severity=DiscrepancySeverity.LOW,
                    stream_a_value=a_desc,
                    stream_b_value=b_desc,
                    ground_truth=None,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=similarity,
                    rationale=f"Description differs (similarity: {similarity:.2f})",
                ))

        # Check parameters
        param_disc = self._detect_parameter_discrepancies(component_id, a_content, b_content)
        discrepancies.extend(param_disc)

        return discrepancies

    def _detect_parameter_discrepancies(
        self,
        component_id: str,
        a_content: dict,
        b_content: dict,
    ) -> list[Discrepancy]:
        """Detect parameter documentation discrepancies."""
        discrepancies = []

        a_params = {p.get("name"): p for p in a_content.get("parameters", [])}
        b_params = {p.get("name"): p for p in b_content.get("parameters", [])}

        all_param_names = set(a_params.keys()) | set(b_params.keys())

        for param_name in all_param_names:
            a_param = a_params.get(param_name)
            b_param = b_params.get(param_name)

            if a_param is None or b_param is None:
                # Missing parameter in one stream
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.PARAMETER_MISMATCH,
                    severity=DiscrepancySeverity.HIGH,
                    stream_a_value=a_param,
                    stream_b_value=b_param,
                    ground_truth=None,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=0.5,
                    rationale=f"Parameter {param_name} missing from one stream",
                ))
            elif a_param.get("type") != b_param.get("type"):
                # Type mismatch
                discrepancies.append(Discrepancy(
                    discrepancy_id=f"disc_{uuid.uuid4().hex[:8]}",
                    component_id=component_id,
                    type=DiscrepancyType.PARAMETER_MISMATCH,
                    severity=DiscrepancySeverity.MEDIUM,
                    stream_a_value=a_param,
                    stream_b_value=b_param,
                    ground_truth=None,
                    resolution=ResolutionType.NEEDS_HUMAN_REVIEW,
                    confidence=0.6,
                    rationale=f"Parameter {param_name} type differs: {a_param.get('type')} vs {b_param.get('type')}",
                ))

        return discrepancies

    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate text similarity (simple implementation)."""
        if not text_a or not text_b:
            return 0.0

        # Normalize
        a_words = set(text_a.lower().split())
        b_words = set(text_b.lower().split())

        if not a_words or not b_words:
            return 0.0

        # Jaccard similarity
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)

        return intersection / union if union > 0 else 0.0
```

---

## 7. Resolution Engine Component

```python
# twinscribe/agents/comparator/resolver.py

from typing import Any

from twinscribe.agents.comparator.models import (
    Discrepancy,
    DiscrepancyType,
    ResolutionType,
)
from twinscribe.analysis.static_oracle import StaticAnalysisOracle


class ResolutionEngine:
    """
    Component responsible for resolving discrepancies using various strategies.

    Resolution Strategies:
    1. Ground Truth: Use static analysis for call graph discrepancies
    2. Preference: Choose stream with better completeness/accuracy
    3. Merge: Combine information from both streams
    4. Escalate: Generate Beads ticket for human review
    """

    def __init__(
        self,
        static_oracle: StaticAnalysisOracle,
        confidence_threshold: float = 0.7,
    ):
        self.static_oracle = static_oracle
        self.confidence_threshold = confidence_threshold

    def resolve_by_ground_truth(self, disc: Discrepancy) -> Discrepancy:
        """
        Resolve call graph discrepancy using static analysis ground truth.

        Decision Logic:
        - If stream_a matches ground_truth: accept_stream_a
        - If stream_b matches ground_truth: accept_stream_b
        - If neither matches: accept_ground_truth (correct both)
        - If both match: should not be a discrepancy
        """
        if disc.ground_truth is None:
            return disc  # Cannot resolve without ground truth

        gt_exists = disc.ground_truth.get("exists", False)

        a_has_value = disc.stream_a_value is not None
        b_has_value = disc.stream_b_value is not None

        if disc.type == DiscrepancyType.CALL_GRAPH_EDGE:
            if gt_exists:
                # Edge should exist
                if a_has_value and not b_has_value:
                    return Discrepancy(
                        **{**disc.model_dump(),
                           "resolution": ResolutionType.ACCEPT_STREAM_A,
                           "confidence": 0.99,
                           "rationale": "Stream A matches ground truth (edge exists)"}
                    )
                elif b_has_value and not a_has_value:
                    return Discrepancy(
                        **{**disc.model_dump(),
                           "resolution": ResolutionType.ACCEPT_STREAM_B,
                           "confidence": 0.99,
                           "rationale": "Stream B matches ground truth (edge exists)"}
                    )
            else:
                # Edge should not exist
                if not a_has_value and b_has_value:
                    return Discrepancy(
                        **{**disc.model_dump(),
                           "resolution": ResolutionType.ACCEPT_STREAM_A,
                           "confidence": 0.99,
                           "rationale": "Stream A matches ground truth (edge does not exist)"}
                    )
                elif a_has_value and not b_has_value:
                    return Discrepancy(
                        **{**disc.model_dump(),
                           "resolution": ResolutionType.ACCEPT_STREAM_B,
                           "confidence": 0.99,
                           "rationale": "Stream B matches ground truth (edge does not exist)"}
                    )

        elif disc.type in (DiscrepancyType.FALSE_CALLEE, DiscrepancyType.MISSING_CALLEE):
            # Ground truth is authoritative
            return Discrepancy(
                **{**disc.model_dump(),
                   "resolution": ResolutionType.ACCEPT_GROUND_TRUTH,
                   "confidence": 0.99,
                   "rationale": f"Static analysis is authoritative for {disc.type.value}"}
            )

        return disc

    def resolve_by_preference(
        self,
        disc: Discrepancy,
        a_metadata: dict,
        b_metadata: dict,
    ) -> Discrepancy:
        """
        Resolve content discrepancy by preferring more complete/confident stream.

        Preference Criteria:
        1. Higher confidence score
        2. More detailed content (length as proxy)
        3. Better formatting/structure
        """
        a_confidence = a_metadata.get("confidence", 0.5)
        b_confidence = b_metadata.get("confidence", 0.5)

        if a_confidence > b_confidence + 0.1:  # Significant difference
            return Discrepancy(
                **{**disc.model_dump(),
                   "resolution": ResolutionType.ACCEPT_STREAM_A,
                   "confidence": a_confidence,
                   "rationale": f"Stream A has higher confidence ({a_confidence:.2f} vs {b_confidence:.2f})"}
            )
        elif b_confidence > a_confidence + 0.1:
            return Discrepancy(
                **{**disc.model_dump(),
                   "resolution": ResolutionType.ACCEPT_STREAM_B,
                   "confidence": b_confidence,
                   "rationale": f"Stream B has higher confidence ({b_confidence:.2f} vs {a_confidence:.2f})"}
            )

        # Similar confidence - check content length as proxy for completeness
        a_len = len(str(disc.stream_a_value)) if disc.stream_a_value else 0
        b_len = len(str(disc.stream_b_value)) if disc.stream_b_value else 0

        if a_len > b_len * 1.5:  # Significantly more content
            return Discrepancy(
                **{**disc.model_dump(),
                   "resolution": ResolutionType.ACCEPT_STREAM_A,
                   "confidence": 0.7,
                   "rationale": "Stream A has more detailed content"}
            )
        elif b_len > a_len * 1.5:
            return Discrepancy(
                **{**disc.model_dump(),
                   "resolution": ResolutionType.ACCEPT_STREAM_B,
                   "confidence": 0.7,
                   "rationale": "Stream B has more detailed content"}
            )

        # Cannot decide with confidence
        return disc

    def resolve_by_merge(self, disc: Discrepancy) -> Discrepancy:
        """
        Attempt to merge information from both streams.

        Applicable for:
        - Descriptions (combine both perspectives)
        - Examples (include all)
        - Notes (concatenate)
        """
        if disc.type not in (
            DiscrepancyType.DESCRIPTION_MISMATCH,
            DiscrepancyType.DOCUMENTATION_CONTENT,
        ):
            return disc  # Merge not applicable

        # Simple merge: concatenate with separator
        a_val = disc.stream_a_value or ""
        b_val = disc.stream_b_value or ""

        merged = f"{a_val}\n\n[Additional perspective:]\n{b_val}"

        return Discrepancy(
            **{**disc.model_dump(),
               "resolution": ResolutionType.MERGE_BOTH,
               "confidence": 0.75,
               "rationale": "Merged content from both streams",
               "ground_truth": merged}  # Store merged value in ground_truth field
        )
```

---

## 8. System Prompt for Comparator

```python
# Addition to twinscribe/agents/prompts.py

COMPARATOR_SYSTEM_PROMPT = """You are the arbitration agent responsible for comparing documentation outputs
from two independent streams and resolving discrepancies.

YOUR RESPONSIBILITIES:
1. Compare outputs component-by-component
2. Identify all discrepancies (structural and semantic)
3. For call graph discrepancies: consult static analysis (ground truth)
4. For documentation content discrepancies: use judgment or escalate
5. Generate Beads tickets for issues requiring human review
6. Track convergence progress

DECISION HIERARCHY:
1. Static analysis is AUTHORITATIVE for call graph accuracy
2. For semantic/content differences, prefer completeness and accuracy
3. When uncertain (confidence < 0.7), mark for human review
4. Never guess - escalate unclear cases

You have access to:
- Stream A validated output
- Stream B validated output
- Static analysis call graph (GROUND TRUTH)
- Component source code (for context)

RESOLUTION OPTIONS:
- accept_stream_a: Stream A's value is correct
- accept_stream_b: Stream B's value is correct
- accept_ground_truth: Use static analysis ground truth
- merge_both: Combine information from both streams
- needs_human_review: Create Beads ticket for review

OUTPUT FORMAT:
Respond with JSON containing:
{
  "resolution": "accept_stream_a|accept_stream_b|merge_both|needs_human_review",
  "confidence": 0.0-1.0,
  "rationale": "Detailed explanation of decision",
  "merged_value": {...}  // Only if resolution is "merge_both"
}

IMPORTANT:
- Be thorough - missing a discrepancy is worse than flagging a false positive
- Document your reasoning clearly for audit purposes
- High-confidence resolutions (>0.7) can be automated
- Low-confidence resolutions require human review
"""
```

---

## 9. Data Flow Diagram

```
                    +-------------------+
                    | Stream A Output   |
                    | (validated)       |
                    +---------+---------+
                              |
                              |     +-------------------+
                              |     | Stream B Output   |
                              |     | (validated)       |
                              |     +---------+---------+
                              |               |
                              v               v
                    +---------+---------------+---------+
                    |       DiscrepancyDetector        |
                    |                                   |
                    | - detect_call_graph_discrepancies |
                    | - detect_content_discrepancies    |
                    +---------+---------------+---------+
                              |
                              v
                    +---------+---------------+---------+
                    |     Static Analysis Oracle       |
                    |     (Ground Truth)               |
                    +---------+---------------+---------+
                              |
                              v
                    +---------+---------------+---------+
                    |      ResolutionEngine            |
                    |                                   |
                    | - resolve_by_ground_truth         |
                    | - resolve_by_preference           |
                    | - resolve_by_merge                |
                    +---------+---------------+---------+
                              |
            +-----------------+-------------------+
            |                 |                   |
            v                 v                   v
    +-------+------+  +-------+------+  +---------+---------+
    | Corrections  |  | Corrections  |  | Beads Tickets      |
    | for Stream A |  | for Stream B |  | (human review)    |
    +--------------+  +--------------+  +-------------------+
            |                 |                   |
            v                 v                   v
    +-------+------+  +-------+------+  +---------+---------+
    | Apply to     |  | Apply to     |  | Create in Beads    |
    | Stream A     |  | Stream B     |  | System             |
    +--------------+  +--------------+  +-------------------+
            |                 |                   |
            +--------+--------+-------------------+
                     |
                     v
            +--------+--------+
            | ConvergenceCriteria |
            |                     |
            | - check_convergence |
            +--------+--------+
                     |
                     v
            +--------+--------+
            | ComparisonResult |
            +------------------+
```

---

## 10. Error Handling

```python
# twinscribe/agents/comparator/exceptions.py

class ComparatorError(Exception):
    """Base exception for comparator-related errors."""
    pass


class DiscrepancyDetectionError(ComparatorError):
    """Raised when discrepancy detection fails."""
    pass


class ResolutionError(ComparatorError):
    """Raised when discrepancy resolution fails."""
    pass


class GroundTruthUnavailableError(ComparatorError):
    """Raised when ground truth is needed but not available."""
    pass


class ConvergenceCheckError(ComparatorError):
    """Raised when convergence check fails."""
    pass


class BeadsTicketCreationError(ComparatorError):
    """Raised when Beads ticket creation fails."""
    pass
```

---

## 11. Interface Contracts Summary

| Interface | Input | Output | Description |
|-----------|-------|--------|-------------|
| `ComparatorAgent.compare()` | `stream_a, stream_b, iteration` | `ComparisonResult` | Main comparison entry point |
| `DiscrepancyDetector.detect_call_graph_discrepancies()` | `component_id, a_doc, b_doc` | `list[Discrepancy]` | Find call graph differences |
| `DiscrepancyDetector.detect_content_discrepancies()` | `component_id, a_doc, b_doc` | `list[Discrepancy]` | Find documentation content differences |
| `ResolutionEngine.resolve_by_ground_truth()` | `Discrepancy` | `Discrepancy` | Resolve using static analysis |
| `ConvergenceCriteria.check_convergence()` | `ComparisonResult, iteration` | `ConvergenceStatus` | Check if converged |

---

## 12. Testing Strategy

### Unit Tests
- Test discrepancy detection with known differences
- Test ground truth resolution with mock oracle
- Test convergence criteria thresholds
- Test Beads ticket generation

### Integration Tests
- Test full comparison pipeline with realistic outputs
- Test correction propagation to streams
- Test convergence over multiple iterations

### Contract Tests
- Verify ComparisonResult schema matches spec
- Verify Beads ticket format is valid
- Test confidence threshold behavior

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-06 | Systems Architect | Initial design |
