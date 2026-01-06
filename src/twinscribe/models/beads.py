"""
Beads ticket models for human review and rebuild tracking.

These models define the ticket structures for Beads integration
as specified in section 5 of the specification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class BeadsTicketPriority(str, Enum):
    """Priority levels for Beads tickets."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class BeadsTicketType(str, Enum):
    """Types of Beads tickets created by the system."""

    CLARIFICATION = "Clarification"  # Discrepancy needing human review
    STORY = "Story"                   # Rebuild ticket
    BUG = "Bug"                       # Issue found in documentation
    TASK = "Task"                     # Follow-up task


class StreamComparison(BaseModel):
    """Comparison of values between streams for ticket display.

    Attributes:
        aspect: What is being compared
        stream_a_value: Value from Stream A
        stream_b_value: Value from Stream B
    """

    aspect: str = Field(..., description="Aspect being compared")
    stream_a_value: Optional[str] = Field(
        default=None, description="Stream A value"
    )
    stream_b_value: Optional[str] = Field(
        default=None, description="Stream B value"
    )


class DiscrepancyTicket(BaseModel):
    """Beads ticket for a documentation discrepancy requiring human review.

    Based on the discrepancy ticket template in spec section 5.1.

    Attributes:
        project: Beads project key
        issue_type: Type of issue (Clarification)
        priority: Ticket priority
        summary: One-line summary
        component_id: Affected component
        file_path: File location
        line_start: Start line
        line_end: End line
        iteration_number: Which iteration found this
        stream_a_model: Model used for Stream A
        stream_b_model: Model used for Stream B
        differences: List of stream value comparisons
        ground_truth_available: Whether static analysis applies
        ground_truth_value: Static analysis value if available
        source_code_snippet: Relevant code snippet
        language: Programming language
        labels: Ticket labels
        custom_fields: Additional custom fields
        discrepancy_id: Internal discrepancy ID
        created_at: When ticket was created
        ticket_key: Beads ticket key once created
    """

    project: str = Field(
        default="LEGACY_DOC",
        description="Beads project key",
    )
    issue_type: BeadsTicketType = Field(
        default=BeadsTicketType.CLARIFICATION,
        description="Issue type",
    )
    priority: BeadsTicketPriority = Field(
        default=BeadsTicketPriority.MEDIUM,
        description="Priority",
    )
    summary: str = Field(
        ...,
        max_length=255,
        description="Ticket summary",
    )
    component_id: str = Field(
        ..., description="Affected component ID"
    )
    file_path: str = Field(
        ..., description="File path"
    )
    line_start: int = Field(
        ..., ge=1, description="Start line"
    )
    line_end: int = Field(
        ..., ge=1, description="End line"
    )
    iteration_number: int = Field(
        default=1, ge=1, description="Iteration found"
    )
    stream_a_model: str = Field(
        ..., description="Stream A model name"
    )
    stream_b_model: str = Field(
        ..., description="Stream B model name"
    )
    differences: list[StreamComparison] = Field(
        default_factory=list,
        description="Value comparisons",
    )
    ground_truth_available: bool = Field(
        default=False,
        description="Static analysis available",
    )
    ground_truth_value: Optional[str] = Field(
        default=None,
        description="Static analysis value",
    )
    source_code_snippet: str = Field(
        default="",
        description="Relevant code",
    )
    language: str = Field(
        default="python",
        description="Programming language",
    )
    labels: list[str] = Field(
        default_factory=lambda: ["ai-documentation"],
        description="Ticket labels",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom field values",
    )
    discrepancy_id: str = Field(
        ..., description="Internal discrepancy ID"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )
    ticket_key: Optional[str] = Field(
        default=None,
        description="Beads ticket key",
    )

    def format_summary(self, discrepancy_type: str) -> str:
        """Format the ticket summary line."""
        return f"[AI-DOC] {discrepancy_type}: {self.component_id}"

    def to_beads_payload(self) -> dict:
        """Convert to Beads API payload format.

        Returns:
            Dict suitable for Beads issue creation
        """
        description = self._render_description()

        return {
            "fields": {
                "project": {"key": self.project},
                "issuetype": {"name": self.issue_type.value},
                "priority": {"name": self.priority.value},
                "summary": self.summary,
                "description": description,
                "labels": self.labels,
                **{k: v for k, v in self.custom_fields.items()},
            }
        }

    def _render_description(self) -> str:
        """Render the full ticket description."""
        lines = [
            "## Discrepancy Summary",
            "",
            f"**Component:** `{self.component_id}`",
            f"**File:** `{self.file_path}:{self.line_start}-{self.line_end}`",
            f"**Iteration:** {self.iteration_number}",
            "",
            "## Stream Comparison",
            "",
            f"| Aspect | Stream A ({self.stream_a_model}) | Stream B ({self.stream_b_model}) |",
            "|--------|-------------------------------|-------------------------------|",
        ]

        for diff in self.differences:
            a_val = diff.stream_a_value or "_none_"
            b_val = diff.stream_b_value or "_none_"
            lines.append(f"| {diff.aspect} | {a_val} | {b_val} |")

        lines.extend([
            "",
            "## Ground Truth Reference",
            "",
        ])

        if self.ground_truth_available:
            lines.append(f"Static analysis indicates: `{self.ground_truth_value}`")
        else:
            lines.append("No static analysis available for this discrepancy type.")

        lines.extend([
            "",
            "## Source Code Context",
            "",
            f"```{self.language}",
            self.source_code_snippet,
            "```",
            "",
            "## Requested Action",
            "",
            "Please review and indicate which interpretation is correct:",
            "- [ ] Stream A is correct",
            "- [ ] Stream B is correct",
            "- [ ] Both are partially correct (provide merged version)",
            "- [ ] Neither is correct (provide correct version)",
            "",
            "## Resolution Notes",
            "",
            "_To be filled by reviewer_",
        ])

        return "\n".join(lines)


class RebuildTicket(BaseModel):
    """Beads ticket for rebuilding a documented component.

    Based on the rebuild ticket template in spec section 5.1.

    Attributes:
        project: Beads project key (REBUILD)
        issue_type: Type of issue (Story)
        priority: Ticket priority
        summary: Component rebuild summary
        component_id: Component being rebuilt
        component_name: Short name
        file_path: Current file location
        line_start: Start line
        line_end: End line
        confidence_score: Documentation confidence (0-100)
        confidence_bucket: Confidence level (high/medium/low)
        documentation_summary: One-line summary
        documentation_description: Full description
        parameters: Parameter documentation
        returns_type: Return type
        returns_description: Return description
        raises: Exception documentation
        callees: Components this calls
        callers: Components that call this
        labels: Ticket labels
        acceptance_criteria: Acceptance criteria list
        created_at: When ticket was created
        ticket_key: Beads ticket key once created
    """

    project: str = Field(
        default="REBUILD",
        description="Beads project key",
    )
    issue_type: BeadsTicketType = Field(
        default=BeadsTicketType.STORY,
        description="Issue type",
    )
    priority: BeadsTicketPriority = Field(
        default=BeadsTicketPriority.MEDIUM,
        description="Priority",
    )
    summary: str = Field(
        ...,
        max_length=255,
        description="Rebuild summary",
    )
    component_id: str = Field(
        ..., description="Component ID"
    )
    component_name: str = Field(
        ..., description="Short name"
    )
    file_path: str = Field(
        ..., description="File path"
    )
    line_start: int = Field(
        ..., ge=1, description="Start line"
    )
    line_end: int = Field(
        ..., ge=1, description="End line"
    )
    confidence_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Confidence percentage",
    )
    confidence_bucket: str = Field(
        default="medium",
        description="Confidence level",
    )
    documentation_summary: str = Field(
        default="", description="One-line summary"
    )
    documentation_description: str = Field(
        default="", description="Full description"
    )
    parameters: list[dict] = Field(
        default_factory=list,
        description="Parameter docs",
    )
    returns_type: Optional[str] = Field(
        default=None, description="Return type"
    )
    returns_description: str = Field(
        default="", description="Return description"
    )
    raises: list[dict] = Field(
        default_factory=list,
        description="Exception docs",
    )
    callees: list[dict] = Field(
        default_factory=list,
        description="Called components",
    )
    callers: list[dict] = Field(
        default_factory=list,
        description="Calling components",
    )
    labels: list[str] = Field(
        default_factory=lambda: ["legacy-rebuild", "ai-documented"],
        description="Ticket labels",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=lambda: [
            "Interface matches documentation exactly",
            "All downstream calls preserved",
            "All upstream integrations maintained",
            "Unit test coverage for documented exceptions",
            "Call graph verified against specification",
        ],
        description="Acceptance criteria",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
    )
    ticket_key: Optional[str] = Field(
        default=None,
        description="Beads ticket key",
    )

    def to_beads_payload(self) -> dict:
        """Convert to Beads API payload format.

        Returns:
            Dict suitable for Beads issue creation
        """
        description = self._render_description()

        # Add confidence bucket to labels
        labels = self.labels.copy()
        labels.append(f"confidence-{self.confidence_bucket}")

        return {
            "fields": {
                "project": {"key": self.project},
                "issuetype": {"name": self.issue_type.value},
                "priority": {"name": self.priority.value},
                "summary": self.summary,
                "description": description,
                "labels": labels,
            }
        }

    def _render_description(self) -> str:
        """Render the full ticket description."""
        lines = [
            "## Component Specification",
            "",
            f"**Current Location:** `{self.file_path}:{self.line_start}-{self.line_end}`",
            f"**Documentation Confidence:** {self.confidence_score}%",
            "**Verified By:** Dual-stream AI analysis + static validation",
            "",
            "## Purpose",
            "",
            self.documentation_summary or "_No summary provided_",
            "",
            "## Detailed Description",
            "",
            self.documentation_description or "_No description provided_",
            "",
            "## Interface Contract",
            "",
            "### Parameters",
            "",
            "| Name | Type | Required | Description |",
            "|------|------|----------|-------------|",
        ]

        for param in self.parameters:
            name = param.get("name", "")
            ptype = param.get("type", "Any")
            required = "Yes" if param.get("required", True) else "No"
            desc = param.get("description", "")
            lines.append(f"| `{name}` | `{ptype}` | {required} | {desc} |")

        if not self.parameters:
            lines.append("| _None_ | | | |")

        lines.extend([
            "",
            "### Returns",
            f"- **Type:** `{self.returns_type or 'None'}`",
            f"- **Description:** {self.returns_description or '_No description_'}",
            "",
            "### Exceptions",
            "",
        ])

        for exc in self.raises:
            etype = exc.get("type", "Exception")
            condition = exc.get("condition", "")
            lines.append(f"- `{etype}`: {condition}")

        if not self.raises:
            lines.append("_None documented_")

        lines.extend([
            "",
            f"## Call Graph",
            "",
            f"### This component calls ({len(self.callees)} dependencies):",
            "",
        ])

        for callee in self.callees:
            cid = callee.get("component_id", "")
            line = callee.get("call_site_line", "?")
            ctype = callee.get("call_type", "direct")
            lines.append(f"- `{cid}` (line {line}) - {ctype}")

        if not self.callees:
            lines.append("_None_")

        lines.extend([
            "",
            f"### Called by ({len(self.callers)} dependents):",
            "",
        ])

        for caller in self.callers:
            cid = caller.get("component_id", "")
            line = caller.get("call_site_line", "?")
            lines.append(f"- `{cid}` (line {line})")

        if not self.callers:
            lines.append("_None_")

        lines.extend([
            "",
            "## Rebuild Checklist",
            "",
            f"- [ ] Implement documented interface exactly",
            f"- [ ] Preserve all {len(self.callees)} downstream dependencies",
            f"- [ ] Ensure compatibility with {len(self.callers)} upstream callers",
            f"- [ ] Add unit tests for documented exceptions",
            f"- [ ] Verify call graph matches specification",
        ])

        return "\n".join(lines)
