"""
Ticket Template Engine.

Renders ticket content from templates for discrepancy and rebuild tickets.
"""

from dataclasses import dataclass, field
from enum import Enum
from string import Template

from pydantic import BaseModel, Field


class TemplateType(str, Enum):
    """Type of ticket template."""

    DISCREPANCY = "discrepancy"
    REBUILD = "rebuild"


@dataclass
class DiscrepancyTemplateData:
    """Data for rendering a discrepancy ticket.

    Attributes:
        discrepancy_id: Unique discrepancy identifier
        component_name: Name of the component with discrepancy
        component_type: Type of component (function, class, etc.)
        file_path: Path to the file containing the component
        discrepancy_type: Type of discrepancy (call_graph, documentation, etc.)
        stream_a_value: Stream A's interpretation
        stream_b_value: Stream B's interpretation
        static_analysis_value: Ground truth from static analysis (if available)
        context: Additional context about the discrepancy
        iteration: Current iteration number
        previous_attempts: Summary of previous resolution attempts
        labels: Labels to apply to the ticket
        priority: Ticket priority
    """

    discrepancy_id: str
    component_name: str
    component_type: str
    file_path: str
    discrepancy_type: str
    stream_a_value: str
    stream_b_value: str
    static_analysis_value: str | None = None
    context: str = ""
    iteration: int = 1
    previous_attempts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    priority: str = "Medium"


@dataclass
class RebuildTemplateData:
    """Data for rendering a rebuild ticket.

    Attributes:
        component_name: Name of the component to rebuild
        component_type: Type of component
        file_path: Current file path
        documentation: Final agreed documentation
        call_graph: Call graph relationships
        dependencies: Component dependencies
        dependents: Components that depend on this one
        complexity_score: Estimated complexity
        rebuild_priority: Priority in rebuild order
        suggested_approach: AI-suggested rebuild approach
        labels: Labels to apply
        epic_key: Parent epic key (if any)
    """

    component_name: str
    component_type: str
    file_path: str
    documentation: str
    call_graph: dict[str, list[str]]  # callers/callees
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    complexity_score: float = 0.0
    rebuild_priority: int = 0
    suggested_approach: str = ""
    labels: list[str] = field(default_factory=list)
    epic_key: str | None = None


@dataclass
class DivergentComponentTemplateData:
    """Data for rendering a divergent component ticket.

    Created when streams don't converge after max_iterations and
    require human review to resolve call graph discrepancies.

    Attributes:
        component_id: Unique component identifier
        component_name: Human-readable component name
        component_type: Type of component (function, class, method)
        file_path: Path to the file containing the component
        stream_a_edges: Call graph edges from Stream A
        stream_b_edges: Call graph edges from Stream B
        edges_only_in_a: Edges present only in Stream A
        edges_only_in_b: Edges present only in Stream B
        common_edges: Edges present in both streams
        iteration_history: Summary of convergence attempts per iteration
        total_iterations: Number of iterations attempted
        final_agreement_rate: Agreement rate at final iteration
        labels: Labels to apply to the ticket
        priority: Ticket priority (default High for divergent components)
    """

    component_id: str
    component_name: str
    component_type: str
    file_path: str
    stream_a_edges: list[tuple[str, str]]  # (caller, callee) tuples
    stream_b_edges: list[tuple[str, str]]
    edges_only_in_a: list[tuple[str, str]]
    edges_only_in_b: list[tuple[str, str]]
    common_edges: list[tuple[str, str]]
    iteration_history: list[dict[str, float | int]] = field(default_factory=list)
    total_iterations: int = 0
    final_agreement_rate: float = 0.0
    labels: list[str] = field(default_factory=list)
    priority: str = "High"


class TicketTemplate(BaseModel):
    """A ticket template definition.

    Attributes:
        name: Template name
        template_type: Type of ticket this template creates
        summary_template: Template for ticket summary
        description_template: Template for ticket description
        default_labels: Default labels to apply
        default_priority: Default priority
    """

    name: str
    template_type: TemplateType
    summary_template: str
    description_template: str
    default_labels: list[str] = Field(default_factory=list)
    default_priority: str = "Medium"


# Default discrepancy ticket template
DEFAULT_DISCREPANCY_TEMPLATE = TicketTemplate(
    name="default_discrepancy",
    template_type=TemplateType.DISCREPANCY,
    summary_template="[Discrepancy] ${component_name}: ${discrepancy_type}",
    description_template="""
## Discrepancy Details

**Component:** ${component_name}
**Type:** ${component_type}
**File:** `${file_path}`
**Discrepancy Type:** ${discrepancy_type}
**Iteration:** ${iteration}

## Stream Interpretations

### Stream A (Anthropic)
```
${stream_a_value}
```

### Stream B (OpenAI)
```
${stream_b_value}
```

${static_analysis_section}

## Context
${context}

${previous_attempts_section}

## Resolution Required

Please review the discrepancy above and provide guidance:

1. **Accept Stream A**: Use Stream A's interpretation
2. **Accept Stream B**: Use Stream B's interpretation
3. **Merge**: Provide a merged interpretation combining both
4. **Manual Override**: Provide the correct interpretation

Add a comment with your resolution in the format:
```
RESOLUTION: <accept_a|accept_b|merge|manual>
<your explanation and/or corrected content>
```
""".strip(),
    default_labels=["ai-documentation", "discrepancy"],
    default_priority="Medium",
)


# Default divergent component ticket template
DEFAULT_DIVERGENT_COMPONENT_TEMPLATE = TicketTemplate(
    name="default_divergent_component",
    template_type=TemplateType.DISCREPANCY,
    summary_template="[Divergent Call Graph] ${component_name}: Streams failed to converge",
    description_template="""
## Divergent Component - Human Review Required

**Component:** ${component_name}
**Type:** ${component_type}
**File:** `${file_path}`
**Component ID:** `${component_id}`

## Convergence Summary

**Total Iterations:** ${total_iterations}
**Final Agreement Rate:** ${final_agreement_rate}%

The dual-stream documentation process could not reach consensus on the call graph
for this component after ${total_iterations} iterations. Human review is required
to determine the correct call relationships.

## Stream A - Call Graph Edges

${stream_a_edges_section}

## Stream B - Call Graph Edges

${stream_b_edges_section}

## Discrepancy Analysis

### Edges Only in Stream A (not found by Stream B)
${edges_only_in_a_section}

### Edges Only in Stream B (not found by Stream A)
${edges_only_in_b_section}

### Common Edges (agreed by both streams)
${common_edges_section}

## Iteration History

${iteration_history_section}

## Resolution Required

Please review the call graph discrepancies and determine the correct edges:

1. **Accept Stream A**: Stream A's call graph is correct
2. **Accept Stream B**: Stream B's call graph is correct
3. **Merge**: Combine edges from both streams as appropriate
4. **Manual Override**: Provide the correct call graph

Add a comment with your resolution in the format:
```
RESOLUTION: <accept_a|accept_b|merge|manual>
EDGES:
- caller1 -> callee1
- caller2 -> callee2
```
""".strip(),
    default_labels=["ai-documentation", "divergent-call-graph", "human-review"],
    default_priority="High",
)


# Default rebuild ticket template
DEFAULT_REBUILD_TEMPLATE = TicketTemplate(
    name="default_rebuild",
    template_type=TemplateType.REBUILD,
    summary_template="[Rebuild] ${component_name} (${component_type})",
    description_template="""
## Component Information

**Name:** ${component_name}
**Type:** ${component_type}
**Current Location:** `${file_path}`
**Rebuild Priority:** ${rebuild_priority}
**Complexity Score:** ${complexity_score}

## Documentation

```
${documentation}
```

## Dependencies

### This component depends on:
${dependencies_list}

### Components that depend on this:
${dependents_list}

## Call Graph

### Calls (outgoing):
${calls_list}

### Called by (incoming):
${called_by_list}

## Suggested Rebuild Approach

${suggested_approach}

## Acceptance Criteria

- [ ] Component rebuilt following documentation
- [ ] All dependencies resolved
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Code review completed
""".strip(),
    default_labels=["ai-documentation", "rebuild"],
    default_priority="Medium",
)


class TicketTemplateEngine:
    """Renders ticket content from templates.

    Supports:
    - Variable substitution using ${variable} syntax
    - Custom templates per ticket type
    - Default templates for common cases

    Usage:
        engine = TicketTemplateEngine()

        # Render discrepancy ticket
        data = DiscrepancyTemplateData(...)
        summary, description = engine.render_discrepancy(data)

        # Render rebuild ticket
        rebuild_data = RebuildTemplateData(...)
        summary, description = engine.render_rebuild(rebuild_data)
    """

    def __init__(self) -> None:
        """Initialize the template engine with default templates."""
        self._templates: dict[str, TicketTemplate] = {
            "default_discrepancy": DEFAULT_DISCREPANCY_TEMPLATE,
            "default_rebuild": DEFAULT_REBUILD_TEMPLATE,
            "default_divergent_component": DEFAULT_DIVERGENT_COMPONENT_TEMPLATE,
        }

    def register_template(self, template: TicketTemplate) -> None:
        """Register a custom template.

        Args:
            template: Template to register
        """
        self._templates[template.name] = template

    def get_template(self, name: str) -> TicketTemplate | None:
        """Get a template by name.

        Args:
            name: Template name

        Returns:
            Template or None if not found
        """
        return self._templates.get(name)

    def render_discrepancy(
        self,
        data: DiscrepancyTemplateData,
        template_name: str = "default_discrepancy",
    ) -> tuple[str, str]:
        """Render a discrepancy ticket.

        Args:
            data: Template data
            template_name: Name of template to use

        Returns:
            Tuple of (summary, description)

        Raises:
            ValueError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        if template.template_type != TemplateType.DISCREPANCY:
            raise ValueError(f"Template {template_name} is not a discrepancy template")

        # Build substitution variables
        variables = {
            "discrepancy_id": data.discrepancy_id,
            "component_name": data.component_name,
            "component_type": data.component_type,
            "file_path": data.file_path,
            "discrepancy_type": data.discrepancy_type,
            "stream_a_value": data.stream_a_value,
            "stream_b_value": data.stream_b_value,
            "context": data.context or "No additional context provided.",
            "iteration": str(data.iteration),
        }

        # Build static analysis section
        if data.static_analysis_value:
            variables["static_analysis_section"] = f"""
### Static Analysis (Ground Truth)
```
{data.static_analysis_value}
```
""".strip()
        else:
            variables["static_analysis_section"] = ""

        # Build previous attempts section
        if data.previous_attempts:
            attempts_text = "\n".join(f"- {attempt}" for attempt in data.previous_attempts)
            variables["previous_attempts_section"] = f"""
## Previous Resolution Attempts
{attempts_text}
""".strip()
        else:
            variables["previous_attempts_section"] = ""

        # Render templates
        summary = Template(template.summary_template).safe_substitute(variables)
        description = Template(template.description_template).safe_substitute(variables)

        return summary, description

    def render_rebuild(
        self,
        data: RebuildTemplateData,
        template_name: str = "default_rebuild",
    ) -> tuple[str, str]:
        """Render a rebuild ticket.

        Args:
            data: Template data
            template_name: Name of template to use

        Returns:
            Tuple of (summary, description)

        Raises:
            ValueError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        if template.template_type != TemplateType.REBUILD:
            raise ValueError(f"Template {template_name} is not a rebuild template")

        # Build substitution variables
        variables = {
            "component_name": data.component_name,
            "component_type": data.component_type,
            "file_path": data.file_path,
            "documentation": data.documentation,
            "rebuild_priority": str(data.rebuild_priority),
            "complexity_score": f"{data.complexity_score:.2f}",
            "suggested_approach": data.suggested_approach or "No specific approach suggested.",
        }

        # Build dependencies list
        if data.dependencies:
            variables["dependencies_list"] = "\n".join(f"* {dep}" for dep in data.dependencies)
        else:
            variables["dependencies_list"] = "* None"

        # Build dependents list
        if data.dependents:
            variables["dependents_list"] = "\n".join(f"* {dep}" for dep in data.dependents)
        else:
            variables["dependents_list"] = "* None"

        # Build call graph lists
        calls = data.call_graph.get("calls", [])
        called_by = data.call_graph.get("called_by", [])

        if calls:
            variables["calls_list"] = "\n".join(f"* {call}" for call in calls)
        else:
            variables["calls_list"] = "* None"

        if called_by:
            variables["called_by_list"] = "\n".join(f"* {caller}" for caller in called_by)
        else:
            variables["called_by_list"] = "* None"

        # Render templates
        summary = Template(template.summary_template).safe_substitute(variables)
        description = Template(template.description_template).safe_substitute(variables)

        return summary, description

    def render_divergent_component(
        self,
        data: DivergentComponentTemplateData,
        template_name: str = "default_divergent_component",
    ) -> tuple[str, str]:
        """Render a divergent component ticket.

        Called when streams fail to converge on call graph after max iterations.

        Args:
            data: Template data for divergent component
            template_name: Name of template to use

        Returns:
            Tuple of (summary, description)

        Raises:
            ValueError: If template not found
        """
        template = self._templates.get(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        # Helper function to format edge list
        def format_edges(edges: list[tuple[str, str]]) -> str:
            if not edges:
                return "* None"
            return "\n".join(f"* `{caller}` -> `{callee}`" for caller, callee in edges)

        # Build substitution variables
        variables = {
            "component_id": data.component_id,
            "component_name": data.component_name,
            "component_type": data.component_type,
            "file_path": data.file_path,
            "total_iterations": str(data.total_iterations),
            "final_agreement_rate": f"{data.final_agreement_rate * 100:.1f}",
        }

        # Format edge sections
        variables["stream_a_edges_section"] = format_edges(data.stream_a_edges)
        variables["stream_b_edges_section"] = format_edges(data.stream_b_edges)
        variables["edges_only_in_a_section"] = format_edges(data.edges_only_in_a)
        variables["edges_only_in_b_section"] = format_edges(data.edges_only_in_b)
        variables["common_edges_section"] = format_edges(data.common_edges)

        # Format iteration history
        if data.iteration_history:
            history_lines = []
            for entry in data.iteration_history:
                iteration = entry.get("iteration", "?")
                rate = entry.get("agreement_rate", 0)
                divergent = entry.get("divergent_count", 0)
                history_lines.append(
                    f"| {iteration} | {rate * 100:.1f}% | {divergent} |"
                )
            variables["iteration_history_section"] = (
                "| Iteration | Agreement Rate | Divergent Components |\n"
                "| --- | --- | --- |\n"
                + "\n".join(history_lines)
            )
        else:
            variables["iteration_history_section"] = "No iteration history available."

        # Render templates
        summary = Template(template.summary_template).safe_substitute(variables)
        description = Template(template.description_template).safe_substitute(variables)

        return summary, description

    def get_labels(
        self,
        data: DiscrepancyTemplateData | RebuildTemplateData | DivergentComponentTemplateData,
        template_name: str,
    ) -> list[str]:
        """Get labels for a ticket.

        Combines template defaults with data-specific labels.

        Args:
            data: Template data
            template_name: Template name

        Returns:
            Combined list of labels
        """
        template = self._templates.get(template_name)
        labels = list(template.default_labels) if template else []

        # Add data-specific labels
        labels.extend(data.labels)

        # Deduplicate while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        return unique_labels

    def get_priority(
        self,
        data: DiscrepancyTemplateData | RebuildTemplateData | DivergentComponentTemplateData,
        template_name: str,
    ) -> str:
        """Get priority for a ticket.

        Uses data priority if specified, otherwise template default.

        Args:
            data: Template data
            template_name: Template name

        Returns:
            Priority string
        """
        # Check if data has priority attribute
        if hasattr(data, "priority") and data.priority:
            return data.priority

        template = self._templates.get(template_name)
        return template.default_priority if template else "Medium"


class ResolutionParser:
    """Parses human resolutions from ticket comments.

    Extracts resolution action and content from structured comments.

    Expected format:
        RESOLUTION: <action>
        <content>

    Where action is one of: accept_a, accept_b, merge, manual
    """

    RESOLUTION_PATTERN = r"RESOLUTION:\s*(accept_a|accept_b|merge|manual)\s*\n?(.*)"

    def __init__(self) -> None:
        """Initialize the parser."""
        import re

        self._pattern = re.compile(self.RESOLUTION_PATTERN, re.IGNORECASE | re.DOTALL)

    def parse(self, comment_text: str) -> tuple[str, str] | None:
        """Parse a resolution from comment text.

        Args:
            comment_text: Raw comment text

        Returns:
            Tuple of (action, content) or None if not a resolution
        """
        match = self._pattern.search(comment_text)
        if not match:
            return None

        action = match.group(1).lower()
        content = match.group(2).strip() if match.group(2) else ""

        return action, content

    def is_resolution_comment(self, comment_text: str) -> bool:
        """Check if comment contains a resolution.

        Args:
            comment_text: Raw comment text

        Returns:
            True if comment contains resolution
        """
        return self._pattern.search(comment_text) is not None

    def extract_action(self, comment_text: str) -> str | None:
        """Extract just the action from a resolution comment.

        Args:
            comment_text: Raw comment text

        Returns:
            Action string or None
        """
        result = self.parse(comment_text)
        return result[0] if result else None

    def extract_content(self, comment_text: str) -> str | None:
        """Extract just the content from a resolution comment.

        Args:
            comment_text: Raw comment text

        Returns:
            Content string or None
        """
        result = self.parse(comment_text)
        return result[1] if result else None


def format_code_block(content: str, language: str = "") -> str:
    """Format content as a Markdown code block.

    Args:
        content: Content to format
        language: Optional language for syntax highlighting

    Returns:
        Markdown-formatted code block
    """
    return f"```{language}\n{content}\n```"


def format_callout(
    content: str,
    title: str = "",
    callout_type: str = "info",
) -> str:
    """Format content as a Markdown callout/blockquote.

    Args:
        content: Callout content
        title: Optional callout title
        callout_type: Callout type (info, warning, error, note)

    Returns:
        Markdown-formatted callout
    """
    prefix = f"> **{callout_type.upper()}**" if not title else f"> **{title}**"
    quoted_content = "\n".join(f"> {line}" for line in content.split("\n"))
    return f"{prefix}\n{quoted_content}"


def format_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    """Format data as a Markdown table.

    Args:
        headers: Table headers
        rows: Table rows

    Returns:
        Markdown-formatted table
    """
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return header_row + "\n" + separator + "\n" + "\n".join(data_rows)
