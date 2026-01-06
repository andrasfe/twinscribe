# Dual-Stream Code Documentation System with Tiered Model Architecture
## Specification v2.0

**Author:** AI Architecture Research  
**Date:** January 2026  
**Status:** Ready for Implementation

---

## 1. Executive Summary

A dual-stream multi-agent system for generating accurate code documentation with call graph linkages. Two independent agent streams document and validate code in parallel, with discrepancies resolved through a premium-tier arbitrator agent that generates Beads tickets. Static analysis serves as the ground truth anchor for call graph validation.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual streams** (not triple) | Static analysis provides ground truth; no need for voting-based tie-breaking |
| **Tiered models** | 90% cost reduction by using cheap models for generation, expensive for judgment |
| **Static analysis anchor** | Call graphs are verifiable against AST; eliminates "consensus of wrong answers" |
| **Beads-based resolution** | Human-in-loop for edge cases; audit trail for decisions |

---

## 2. Architecture

### 2.1 System Diagram

```
                            ┌─────────────────────────────────┐
                            │         SOURCE CODE             │
                            │    + Static Analysis Tools      │
                            │    (PyCG, pyan3, java-callgraph)│
                            └───────────────┬─────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
┌───────────────────────────┐   ┌─────────────────┐   ┌───────────────────────────┐
│        STREAM A           │   │  GROUND TRUTH   │   │        STREAM B           │
│                           │   │                 │   │                           │
│  ┌─────────────────────┐  │   │  Static Call    │   │  ┌─────────────────────┐  │
│  │ A1: Documenter      │  │   │  Graph (AST)    │   │  │ B1: Documenter      │  │
│  │ Claude Sonnet 4.5   │  │   │                 │   │  │ GPT-4o              │  │
│  │ $3/M tokens         │  │   │  Extracted via: │   │  │ $2.50/M tokens      │  │
│  └──────────┬──────────┘  │   │  • PyCG         │   │  └──────────┬──────────┘  │
│             │             │   │  • pyan3        │   │             │             │
│             ▼             │   │  • Sourcetrail  │   │             ▼             │
│  ┌─────────────────────┐  │   │                 │   │  ┌─────────────────────┐  │
│  │ A2: Validator       │  │   └────────┬────────┘   │  │ B2: Validator       │  │
│  │ Claude Haiku 4.5    │  │            │            │  │ GPT-4o-mini         │  │
│  │ $0.25/M tokens      │  │            │            │  │ $0.15/M tokens      │  │
│  └──────────┬──────────┘  │            │            │  └──────────┬──────────┘  │
│             │             │            │            │             │             │
└─────────────┼─────────────┘            │            └─────────────┼─────────────┘
              │                          │                          │
              │         ┌────────────────┼────────────────┐         │
              │         │                │                │         │
              └────────►│    ┌───────────▼───────────┐    │◄────────┘
                        │    │                       │    │
                        │    │   C: COMPARATOR       │    │
                        │    │   Claude Opus 4.5     │    │
                        │    │   $15/M tokens        │    │
                        │    │                       │    │
                        │    │   • Compare outputs   │    │
                        │    │   • Consult ground    │    │
                        │    │     truth for ties    │    │
                        │    │   • Generate Beadss    │    │
                        │    │   • Verify convergence│    │
                        │    │                       │    │
                        │    └───────────┬───────────┘    │
                        │                │                │
                        └────────────────┼────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
           ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
           │ Converged    │    │ Beads Tickets │    │ Final Merged │
           │ (iterate)    │    │ (resolve)    │    │ Documentation│
           └──────────────┘    └──────────────┘    └──────────────┘
```

### 2.2 Model Tier Specification

#### API keys in .env and provide a .env.example. Will use OPENROUTER for this.

| Tier | Models | Cost | Use Case | Token Volume |
|------|--------|------|----------|--------------|
| **Generation** | Claude Sonnet 4.5, GPT-4o | ~$3/M | Documentation writing | High |
| **Validation** | Claude Haiku 4.5, GPT-4o-mini | ~$0.20/M | Verification checks | High |
| **Arbitration** | Claude Opus 4.5 | ~$15/M | Comparison, judgment, Beads | Low |

**Cost Projection (1000 components):**
```
Generation:  2 streams × 500K tokens × $3/M    = $3.00
Validation:  2 streams × 500K tokens × $0.20/M = $0.20
Arbitration: 1 agent   × 100K tokens × $15/M   = $1.50
─────────────────────────────────────────────────────
Total per iteration:                           ≈ $4.70
vs. All-Opus baseline:                         ≈ $45.00
Savings:                                         ~90%
```

---

## 3. Agent Specifications

### 3.1 Documenter Agents (A1, B1)

**Model Tier:** Generation (Claude Sonnet 4.5 / GPT-4o)

**Purpose:** Generate comprehensive documentation with call graph linkages for each code component.

**Input:**
- Source code component (function, method, class)
- Dependency context (already-documented dependencies)
- Static analysis hints (optional, for guidance)

**Output Schema:**
```json
{
  "component_id": "module.Class.method",
  "documentation": {
    "summary": "One-line description",
    "description": "Detailed explanation of purpose and behavior",
    "parameters": [
      {
        "name": "param_name",
        "type": "str",
        "description": "What this parameter does",
        "default": null
      }
    ],
    "returns": {
      "type": "ReturnType",
      "description": "What is returned and when"
    },
    "raises": [
      {
        "type": "ValueError",
        "condition": "When triggered"
      }
    ],
    "examples": ["usage_example()"]
  },
  "call_graph": {
    "callers": [
      {
        "component_id": "module.Other.caller",
        "call_site_line": 45,
        "call_type": "direct|conditional|loop"
      }
    ],
    "callees": [
      {
        "component_id": "module.Helper.helper",
        "call_site_line": 12,
        "call_type": "direct"
      }
    ]
  },
  "metadata": {
    "agent_id": "A1",
    "model": "claude-sonnet-4-5",
    "timestamp": "2026-01-06T10:00:00Z",
    "confidence": 0.92,
    "processing_order": 42
  }
}
```

**Processing Order:** Topological (dependencies first)

**System Prompt Template:**
```
You are a code documentation agent. Your task is to generate comprehensive 
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
that don't exist in the code.
```

### 3.2 Validator Agents (A2, B2)

**Model Tier:** Validation (Claude Haiku 4.5 / GPT-4o-mini)

**Purpose:** Verify documentation completeness and call graph accuracy against source code.

**Input:**
- Documentation output from Documenter
- Original source code
- Static analysis call graph (ground truth)

**Output Schema:**
```json
{
  "component_id": "module.Class.method",
  "validation_result": "pass|fail|warning",
  "completeness": {
    "score": 0.95,
    "missing_elements": ["exception: RuntimeError not documented"],
    "extra_elements": []
  },
  "call_graph_accuracy": {
    "score": 0.98,
    "verified_callees": ["module.Helper.helper"],
    "missing_callees": [],
    "false_callees": [],
    "verified_callers": ["module.Other.caller"],
    "missing_callers": [],
    "false_callers": []
  },
  "corrections_applied": [
    {
      "field": "call_graph.callees",
      "action": "removed",
      "value": "nonexistent.function",
      "reason": "Not found in static analysis"
    }
  ],
  "metadata": {
    "agent_id": "A2",
    "model": "claude-haiku-4-5",
    "static_analyzer": "pycg",
    "timestamp": "2026-01-06T10:01:00Z"
  }
}
```

**Validation Checks:**

| Check | Method | Fail Threshold |
|-------|--------|----------------|
| Parameter completeness | AST parameter extraction | Any missing |
| Return type documented | AST return analysis | If returns value |
| Callees accuracy | Static analysis comparison | >5% false positive |
| Callers accuracy | Static analysis comparison | >10% missing |
| Docstring format | Schema validation | Invalid JSON |

**System Prompt Template:**
```
You are a documentation validation agent. Your task is to verify that 
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

If you find discrepancies between documented call graph and static analysis:
- Trust static analysis
- Flag the discrepancy
- Apply correction

Output validation results in the specified JSON schema.
```

### 3.3 Comparator Agent (C)

**Model Tier:** Arbitration (Claude Opus 4.5)

**Purpose:** Compare outputs from both streams, identify discrepancies, consult ground truth, generate Beads tickets for unresolved issues, verify convergence.

**Input:**
- Validated output from Stream A
- Validated output from Stream B  
- Static analysis call graph (ground truth)
- Previous iteration context (if any)

**Output Schema:**
```json
{
  "comparison_id": "cmp_20260106_001",
  "iteration": 1,
  "summary": {
    "total_components": 247,
    "identical": 235,
    "discrepancies": 12,
    "resolved_by_ground_truth": 8,
    "requires_human_review": 4
  },
  "discrepancies": [
    {
      "discrepancy_id": "disc_001",
      "component_id": "module.Class.method",
      "type": "call_graph_edge",
      "stream_a_value": {"callee": "helper.process", "line": 45},
      "stream_b_value": null,
      "ground_truth": {"callee": "helper.process", "line": 45},
      "resolution": "accept_stream_a",
      "confidence": 0.99,
      "requires_Beads": false
    },
    {
      "discrepancy_id": "disc_002",
      "component_id": "module.Other.func",
      "type": "documentation_content",
      "stream_a_value": "Processes input data synchronously",
      "stream_b_value": "Handles data processing with async fallback",
      "ground_truth": null,
      "resolution": "needs_human_review",
      "confidence": 0.45,
      "requires_Beads": true,
      "Beads_ticket": {
        "summary": "[AI-DOC] Documentation discrepancy: module.Other.func",
        "description": "...",
        "priority": "Medium"
      }
    }
  ],
  "convergence_status": {
    "converged": false,
    "blocking_discrepancies": 4,
    "recommendation": "generate_Beads_tickets"
  },
  "metadata": {
    "agent_id": "C",
    "model": "claude-opus-4-5",
    "timestamp": "2026-01-06T10:02:00Z",
    "comparison_duration_ms": 4500
  }
}
```

**Decision Logic:**

```
FOR each component:
    IF stream_a == stream_b:
        → ACCEPT (identical)
    
    ELSE IF discrepancy is call_graph_related:
        → CONSULT static analysis ground truth
        → ACCEPT matching stream
        → LOG correction to non-matching stream
    
    ELSE IF discrepancy is documentation_content:
        IF one stream has more complete/accurate content (high confidence):
            → ACCEPT better stream
        ELSE:
            → GENERATE Beads ticket for human review
    
    ELSE:
        → GENERATE Beads ticket for human review
```

**System Prompt Template:**
```
You are the arbitration agent responsible for comparing documentation outputs 
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
3. When uncertain (confidence < 0.7), generate Beads ticket
4. Never guess - escalate unclear cases

You have access to:
- Stream A validated output
- Stream B validated output  
- Static analysis call graph (GROUND TRUTH)
- Component source code (for context)

Output comparison results in the specified JSON schema.
Be thorough - missing a discrepancy is worse than flagging a false positive.
```

---

## 4. Workflow

### 4.1 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INITIALIZATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Parse codebase → AST                                                    │
│  2. Build dependency graph                                                  │
│  3. Run static analysis → Ground truth call graph                           │
│  4. Topological sort → Processing order                                     │
│  5. Initialize both streams                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ITERATION LOOP                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: PARALLEL DOCUMENTATION                                       │   │
│  │                                                                       │   │
│  │   Stream A                           Stream B                         │   │
│  │   ────────                           ────────                         │   │
│  │   FOR each component (topo order):   FOR each component (topo order): │   │
│  │     A1: Generate documentation         B1: Generate documentation     │   │
│  │     A2: Validate against static        B2: Validate against static    │   │
│  │                                                                       │   │
│  │   [Parallel execution - both streams run simultaneously]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: COMPARISON                                                   │   │
│  │                                                                       │   │
│  │   Agent C compares Stream A vs Stream B                               │   │
│  │   - Structural comparison (call graphs)                               │   │
│  │   - Semantic comparison (documentation content)                       │   │
│  │   - Ground truth consultation for call graph ties                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: RESOLUTION                                                   │   │
│  │                                                                       │   │
│  │   IF converged (no blocking discrepancies):                          │   │
│  │     → EXIT LOOP → Produce final output                               │   │
│  │                                                                       │   │
│  │   ELSE IF discrepancies resolvable by ground truth:                  │   │
│  │     → Apply corrections                                              │   │
│  │     → Continue to next iteration                                     │   │
│  │                                                                       │   │
│  │   ELSE:                                                              │   │
│  │     → Generate Beads tickets                                          │   │
│  │     → WAIT for ticket resolution                                     │   │
│  │     → Apply resolutions                                              │   │
│  │     → Continue to next iteration                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: ITERATION CHECK                                              │   │
│  │                                                                       │   │
│  │   IF iteration >= MAX_ITERATIONS (5):                                │   │
│  │     → Force convergence with remaining discrepancies logged          │   │
│  │     → EXIT LOOP                                                      │   │
│  │                                                                       │   │
│  │   ELSE:                                                              │   │
│  │     → INCREMENT iteration                                            │   │
│  │     → RETURN to Step 1 (only re-process changed components)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINALIZATION                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Merge converged outputs                                                 │
│  2. Generate final documentation package                                    │
│  3. Generate Beads-ready rebuild tickets                                     │
│  4. Produce convergence report                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Convergence Criteria

```python
class ConvergenceCriteria:
    # Hard thresholds
    MAX_ITERATIONS = 5
    
    # Soft thresholds (converged if ALL met)
    CALL_GRAPH_MATCH_RATE = 0.98      # 98% edges identical
    DOCUMENTATION_SIMILARITY = 0.95   # 95% semantic similarity
    MAX_OPEN_DISCREPANCIES = 2        # Max unresolved non-blocking issues
    
    # Blocking conditions (prevent convergence)
    BLOCKING_DISCREPANCY_TYPES = [
        "missing_critical_call",       # Call exists in code but undocumented
        "false_critical_call",         # Documented call doesn't exist
        "missing_public_api_doc",      # Public method undocumented
        "security_relevant_gap"        # Security-sensitive code undocumented
    ]
```

---

## 5. Beads Integration

### 5.1 Ticket Templates

**Discrepancy Ticket:**
```yaml
project: LEGACY_DOC
issue_type: Clarification
priority: Medium

summary: "[AI-DOC] {{discrepancy_type}}: {{component_id}}"

description: |
  ## Discrepancy Summary
  
  **Component:** `{{component_id}}`
  **File:** `{{file_path}}:{{line_start}}-{{line_end}}`
  **Iteration:** {{iteration_number}}
  
  ## Stream Comparison
  
  | Aspect | Stream A ({{stream_a_model}}) | Stream B ({{stream_b_model}}) |
  |--------|-------------------------------|-------------------------------|
  {{#differences}}
  | {{aspect}} | {{a_value}} | {{b_value}} |
  {{/differences}}
  
  ## Ground Truth Reference
  
  {{#if ground_truth_available}}
  Static analysis indicates: `{{ground_truth_value}}`
  {{else}}
  No static analysis available for this discrepancy type.
  {{/if}}
  
  ## Source Code Context
  
  ```{{language}}
  {{source_code_snippet}}
  ```
  
  ## Requested Action
  
  Please review and indicate which interpretation is correct:
  - [ ] Stream A is correct
  - [ ] Stream B is correct
  - [ ] Both are partially correct (provide merged version)
  - [ ] Neither is correct (provide correct version)
  
  ## Resolution Notes
  
  _To be filled by reviewer_

labels:
  - ai-documentation
  - {{discrepancy_type}}
  - iteration-{{iteration_number}}

custom_fields:
  cf_component_id: "{{component_id}}"
  cf_stream_a_confidence: {{stream_a_confidence}}
  cf_stream_b_confidence: {{stream_b_confidence}}
  cf_auto_resolvable: {{auto_resolvable}}
```

**Rebuild Ticket (Final Output):**
```yaml
project: REBUILD
issue_type: Story
priority: Medium

summary: "Rebuild: {{component_name}}"

description: |
  ## Component Specification
  
  **Current Location:** `{{file_path}}:{{line_start}}-{{line_end}}`
  **Documentation Confidence:** {{confidence_score}}%
  **Verified By:** Dual-stream AI analysis + static validation
  
  ## Purpose
  
  {{documentation.summary}}
  
  ## Detailed Description
  
  {{documentation.description}}
  
  ## Interface Contract
  
  ### Parameters
  {{#each parameters}}
  | Name | Type | Required | Description |
  |------|------|----------|-------------|
  | `{{name}}` | `{{type}}` | {{#if required}}Yes{{else}}No{{/if}} | {{description}} |
  {{/each}}
  
  ### Returns
  - **Type:** `{{returns.type}}`
  - **Description:** {{returns.description}}
  
  ### Exceptions
  {{#each raises}}
  - `{{type}}`: {{condition}}
  {{/each}}
  
  ## Call Graph
  
  ### This component calls ({{callees.length}} dependencies):
  {{#each callees}}
  - `{{component_id}}` (line {{call_site_line}}) - {{call_type}}
  {{/each}}
  
  ### Called by ({{callers.length}} dependents):
  {{#each callers}}
  - `{{component_id}}` (line {{call_site_line}})
  {{/each}}
  
  ## Rebuild Checklist
  
  - [ ] Implement documented interface exactly
  - [ ] Preserve all {{callees.length}} downstream dependencies
  - [ ] Ensure compatibility with {{callers.length}} upstream callers
  - [ ] Add unit tests for documented exceptions
  - [ ] Verify call graph matches specification

acceptance_criteria:
  - Interface matches documentation exactly
  - All downstream calls preserved
  - All upstream integrations maintained  
  - Unit test coverage for documented exceptions
  - Call graph verified against specification

labels:
  - legacy-rebuild
  - ai-documented
  - confidence-{{confidence_bucket}}
```

### 5.2 Ticket Lifecycle Automation

```python
class BeadsLifecycleManager:
    """Manages Beads ticket creation and resolution monitoring."""
    
    async def create_discrepancy_ticket(self, discrepancy: Discrepancy) -> str:
        """Create a Beads ticket for a discrepancy requiring human review."""
        ticket = await self.Beads.create_issue(
            project="LEGACY_DOC",
            issue_type="Clarification",
            summary=f"[AI-DOC] {discrepancy.type}: {discrepancy.component_id}",
            description=self.render_template("discrepancy", discrepancy),
            labels=["ai-documentation", discrepancy.type],
            custom_fields={
                "cf_component_id": discrepancy.component_id,
                "cf_iteration": self.current_iteration
            }
        )
        
        # Register for monitoring
        await self.tracker.register(ticket.key, discrepancy.id)
        
        return ticket.key
    
    async def wait_for_resolution(self, ticket_keys: List[str], timeout_hours: int = 48):
        """Wait for tickets to be resolved, with timeout."""
        start = datetime.now()
        
        while True:
            open_tickets = await self.get_open_tickets(ticket_keys)
            
            if not open_tickets:
                return ResolutionResult(all_resolved=True)
            
            if (datetime.now() - start).hours > timeout_hours:
                return ResolutionResult(
                    all_resolved=False,
                    pending=open_tickets,
                    recommendation="proceed_with_partial"
                )
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def apply_resolution(self, ticket_key: str) -> None:
        """Apply the resolution from a closed ticket."""
        ticket = await self.Beads.get_issue(ticket_key)
        
        resolution = self.parse_resolution(ticket.description, ticket.comments)
        discrepancy = await self.tracker.get_discrepancy(ticket_key)
        
        if resolution.chosen_stream == "A":
            await self.stream_b.apply_correction(
                discrepancy.component_id, 
                resolution.correct_value
            )
        elif resolution.chosen_stream == "B":
            await self.stream_a.apply_correction(
                discrepancy.component_id,
                resolution.correct_value
            )
        else:
            # Custom resolution provided
            await self.stream_a.apply_correction(
                discrepancy.component_id,
                resolution.merged_value
            )
            await self.stream_b.apply_correction(
                discrepancy.component_id,
                resolution.merged_value
            )
```

---

## 6. Static Analysis Integration

### 6.1 Tool Configuration

```yaml
# static_analysis_config.yaml

python:
  primary: pycg
  fallback: pyan3
  config:
    pycg:
      max_iter: 5
      operation: call-graph
      output_format: json
    pyan3:
      no_defines: true
      no_uses: false
      format: json

java:
  primary: java-callgraph-static
  fallback: wala
  config:
    java-callgraph:
      output: json
    wala:
      analysis: 0-CFA

javascript:
  primary: typescript-call-graph
  config:
    include_node_modules: false
    output: json

multi_language:
  fallback: sourcetrail
  config:
    index_path: ./sourcetrail_index
```

### 6.2 Ground Truth Extraction

```python
class StaticAnalysisOracle:
    """Provides ground truth call graph from static analysis."""
    
    def __init__(self, codebase_path: str, language: str):
        self.codebase_path = codebase_path
        self.analyzer = self._get_analyzer(language)
        self._call_graph = None
    
    def _get_analyzer(self, language: str):
        analyzers = {
            "python": PyCGAnalyzer(),
            "java": JavaCallGraphAnalyzer(),
            "javascript": TSCallGraphAnalyzer(),
            "multi": SourcetrailAnalyzer()
        }
        return analyzers.get(language, analyzers["multi"])
    
    @cached_property
    def call_graph(self) -> CallGraph:
        """Extract and cache the ground truth call graph."""
        if self._call_graph is None:
            raw = self.analyzer.analyze(self.codebase_path)
            self._call_graph = self._normalize(raw)
        return self._call_graph
    
    def get_callees(self, component_id: str) -> List[CallEdge]:
        """Get all functions/methods called by a component."""
        return [
            edge for edge in self.call_graph.edges
            if edge.caller == component_id
        ]
    
    def get_callers(self, component_id: str) -> List[CallEdge]:
        """Get all functions/methods that call a component."""
        return [
            edge for edge in self.call_graph.edges
            if edge.callee == component_id
        ]
    
    def verify_edge(self, caller: str, callee: str) -> bool:
        """Verify if a call relationship exists."""
        return any(
            e.caller == caller and e.callee == callee
            for e in self.call_graph.edges
        )
    
    def diff_against(self, documented_graph: CallGraph) -> CallGraphDiff:
        """Compare documented call graph against ground truth."""
        truth_edges = set((e.caller, e.callee) for e in self.call_graph.edges)
        doc_edges = set((e.caller, e.callee) for e in documented_graph.edges)
        
        return CallGraphDiff(
            missing_in_doc=truth_edges - doc_edges,
            extra_in_doc=doc_edges - truth_edges,
            matching=truth_edges & doc_edges,
            precision=len(truth_edges & doc_edges) / len(doc_edges) if doc_edges else 1.0,
            recall=len(truth_edges & doc_edges) / len(truth_edges) if truth_edges else 1.0
        )
```

---

## 7. Implementation

### 7.1 Main Orchestrator

```python
#!/usr/bin/env python3
"""
Dual-Stream Documentation System - Main Orchestrator
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ModelTier(Enum):
    GENERATION = "generation"
    VALIDATION = "validation"
    ARBITRATION = "arbitration"

@dataclass
class ModelConfig:
    tier: ModelTier
    provider: str
    model_name: str
    cost_per_million: float

# Model configurations
MODELS = {
    "stream_a_documenter": ModelConfig(
        ModelTier.GENERATION, "anthropic", "claude-sonnet-4-5-20250929", 3.0
    ),
    "stream_a_validator": ModelConfig(
        ModelTier.VALIDATION, "anthropic", "claude-haiku-4-5-20251001", 0.25
    ),
    "stream_b_documenter": ModelConfig(
        ModelTier.GENERATION, "openai", "gpt-4o", 2.50
    ),
    "stream_b_validator": ModelConfig(
        ModelTier.VALIDATION, "openai", "gpt-4o-mini", 0.15
    ),
    "comparator": ModelConfig(
        ModelTier.ARBITRATION, "anthropic", "claude-opus-4-5-20251101", 15.0
    ),
}


class DualStreamOrchestrator:
    """Main orchestrator for the dual-stream documentation system."""
    
    def __init__(
        self,
        codebase_path: str,
        language: str,
        Beads_config: dict,
        max_iterations: int = 5
    ):
        self.codebase_path = codebase_path
        self.language = language
        self.max_iterations = max_iterations
        
        # Initialize components
        self.static_oracle = StaticAnalysisOracle(codebase_path, language)
        self.Beads_manager = BeadsLifecycleManager(Beads_config)
        
        # Initialize streams
        self.stream_a = DocumentationStream(
            stream_id="A",
            documenter_model=MODELS["stream_a_documenter"],
            validator_model=MODELS["stream_a_validator"],
            static_oracle=self.static_oracle
        )
        self.stream_b = DocumentationStream(
            stream_id="B",
            documenter_model=MODELS["stream_b_documenter"],
            validator_model=MODELS["stream_b_validator"],
            static_oracle=self.static_oracle
        )
        
        # Initialize comparator
        self.comparator = ComparatorAgent(
            model=MODELS["comparator"],
            static_oracle=self.static_oracle
        )
        
        # State
        self.iteration = 0
        self.convergence_history = []
    
    async def run(self) -> DocumentationPackage:
        """Execute the full documentation pipeline."""
        
        # Step 0: Initialize
        print("Initializing...")
        components = await self._discover_components()
        processing_order = self._topological_sort(components)
        
        print(f"Found {len(components)} components to document")
        print(f"Ground truth call graph: {len(self.static_oracle.call_graph.edges)} edges")
        
        # Iteration loop
        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n=== Iteration {self.iteration} ===")
            
            # Step 1: Parallel documentation
            print("Running parallel documentation streams...")
            output_a, output_b = await asyncio.gather(
                self.stream_a.process(processing_order),
                self.stream_b.process(processing_order)
            )
            
            # Step 2: Comparison
            print("Comparing outputs...")
            comparison = await self.comparator.compare(output_a, output_b)
            self.convergence_history.append(comparison.summary)
            
            print(f"  Identical: {comparison.summary['identical']}")
            print(f"  Discrepancies: {comparison.summary['discrepancies']}")
            print(f"  Resolved by ground truth: {comparison.summary['resolved_by_ground_truth']}")
            print(f"  Requires human review: {comparison.summary['requires_human_review']}")
            
            # Step 3: Check convergence
            if comparison.convergence_status['converged']:
                print("\n✓ Streams converged!")
                break
            
            # Step 4: Handle discrepancies
            if comparison.summary['requires_human_review'] > 0:
                print(f"\nGenerating {comparison.summary['requires_human_review']} Beads tickets...")
                
                ticket_keys = []
                for disc in comparison.discrepancies:
                    if disc['requires_Beads']:
                        key = await self.Beads_manager.create_discrepancy_ticket(disc)
                        ticket_keys.append(key)
                        print(f"  Created: {key}")
                
                # Wait for resolution
                print("\nWaiting for ticket resolution...")
                resolution_result = await self.Beads_manager.wait_for_resolution(ticket_keys)
                
                if resolution_result.all_resolved:
                    print("All tickets resolved, applying corrections...")
                    for key in ticket_keys:
                        await self.Beads_manager.apply_resolution(key)
                else:
                    print(f"Timeout: {len(resolution_result.pending)} tickets still open")
            
            # Apply ground-truth corrections
            for disc in comparison.discrepancies:
                if disc['resolution'].startswith('accept_stream_'):
                    await self._apply_correction(disc)
        
        # Step 5: Finalize
        print("\n=== Finalizing ===")
        final_output = await self._merge_outputs(output_a, output_b, comparison)
        
        # Generate rebuild tickets
        rebuild_tickets = await self._generate_rebuild_tickets(final_output)
        
        return DocumentationPackage(
            documentation=final_output,
            call_graph=self._extract_final_call_graph(final_output),
            rebuild_tickets=rebuild_tickets,
            convergence_report=self._generate_convergence_report()
        )
    
    async def _discover_components(self) -> List[Component]:
        """Discover all documentable components in the codebase."""
        # Implementation: AST parsing to find functions, methods, classes
        pass
    
    def _topological_sort(self, components: List[Component]) -> List[Component]:
        """Sort components so dependencies are processed first."""
        # Implementation: Topological sort using dependency graph
        pass
    
    async def _apply_correction(self, discrepancy: dict) -> None:
        """Apply a correction based on comparison result."""
        if discrepancy['resolution'] == 'accept_stream_a':
            await self.stream_b.apply_correction(
                discrepancy['component_id'],
                discrepancy['stream_a_value']
            )
        elif discrepancy['resolution'] == 'accept_stream_b':
            await self.stream_a.apply_correction(
                discrepancy['component_id'],
                discrepancy['stream_b_value']
            )
    
    async def _merge_outputs(self, output_a, output_b, comparison) -> dict:
        """Merge the two stream outputs into final documentation."""
        # Implementation: Merge logic based on comparison results
        pass
    
    async def _generate_rebuild_tickets(self, final_output: dict) -> List[dict]:
        """Generate Beads tickets for rebuilding each component."""
        # Implementation: Create rebuild ticket for each component
        pass
    
    def _generate_convergence_report(self) -> dict:
        """Generate a report on the convergence process."""
        return {
            "total_iterations": self.iteration,
            "history": self.convergence_history,
            "final_status": "converged" if self.iteration < self.max_iterations else "max_iterations_reached"
        }


# Entry point
async def main():
    orchestrator = DualStreamOrchestrator(
        codebase_path="/path/to/legacy/codebase",
        language="python",
        Beads_config={
            "server": "https://your-org.atlassian.net",
            "project": "LEGACY_DOC",
            "username": "automation@your-org.com",
            "api_token": "${Beads_API_TOKEN}"
        },
        max_iterations=5
    )
    
    result = await orchestrator.run()
    
    # Save outputs
    with open("documentation.json", "w") as f:
        json.dump(result.documentation, f, indent=2)
    
    with open("call_graph.json", "w") as f:
        json.dump(result.call_graph, f, indent=2)
    
    print(f"\nDocumentation complete!")
    print(f"  Components documented: {len(result.documentation['components'])}")
    print(f"  Call graph edges: {len(result.call_graph['edges'])}")
    print(f"  Rebuild tickets generated: {len(result.rebuild_tickets)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 7.2 Dependencies

```toml
# pyproject.toml

[project]
name = "dual-stream-documenter"
version = "2.0.0"
requires-python = ">=3.11"

dependencies = [
    # LLM clients
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    
    # Agent orchestration
    "langgraph>=0.2.0",
    
    # Static analysis
    "pycg>=0.0.7",
    "pyan3>=1.2.0",
    
    # Beads integration
    "Beads>=3.8.0",
    "atlassian-python-api>=3.41.0",
    
    # Utilities
    "pydantic>=2.0.0",
    "aiohttp>=3.9.0",
    "tenacity>=8.2.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
java = ["javalang>=0.13.0"]
js = ["esprima>=4.0.0"]
```

---

## 8. Metrics & Monitoring

### 8.1 Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Convergence Rate** | % of runs that converge within max iterations | >95% |
| **Iterations to Converge** | Average iterations needed | <3 |
| **Call Graph Precision** | Documented edges that exist in ground truth | >98% |
| **Call Graph Recall** | Ground truth edges that are documented | >95% |
| **Beads Ticket Rate** | % of components requiring human review | <5% |
| **Cost per Component** | Average cost to document one component | <$0.05 |

### 8.2 Logging Schema

```json
{
  "run_id": "run_20260106_001",
  "timestamp": "2026-01-06T10:00:00Z",
  "codebase": "/path/to/codebase",
  "metrics": {
    "components_total": 247,
    "iterations": 2,
    "convergence_status": "converged",
    "call_graph": {
      "precision": 0.987,
      "recall": 0.962,
      "f1": 0.974
    },
    "discrepancies": {
      "total": 15,
      "resolved_by_ground_truth": 12,
      "resolved_by_Beads": 3,
      "unresolved": 0
    },
    "cost": {
      "stream_a": 2.45,
      "stream_b": 1.89,
      "comparator": 1.12,
      "total": 5.46
    },
    "duration_seconds": 1847
  }
}
```

---

## 9. Appendices

### A. Model Pricing Reference (as of Jan 2026)

| Model | Input ($/M) | Output ($/M) | Context | Best For |
|-------|-------------|--------------|---------|----------|
| Claude Opus 4.5 | $15 | $75 | 200K | Complex reasoning, arbitration |
| Claude Sonnet 4.5 | $3 | $15 | 200K | Balanced generation |
| Claude Haiku 4.5 | $0.25 | $1.25 | 200K | Fast validation |
| GPT-4o | $2.50 | $10 | 128K | Alternative generation |
| GPT-4o-mini | $0.15 | $0.60 | 128K | Cheap validation |

### B. Static Analysis Tool Comparison

| Tool | Language | Output | Accuracy | Speed |
|------|----------|--------|----------|-------|
| PyCG | Python | JSON | High | Fast |
| pyan3 | Python | JSON/Graphviz | Medium | Fast |
| java-callgraph | Java | Text | High | Medium |
| WALA | Java | Various | Very High | Slow |
| TypeScript Call Graph | JS/TS | JSON | Medium | Fast |
| Sourcetrail | Multi | Database | High | Slow |

### C. Configuration Template

```yaml
# config.yaml

codebase:
  path: /path/to/legacy/codebase
  language: python
  exclude_patterns:
    - "**/test_*"
    - "**/tests/**"
    - "**/__pycache__/**"

models:
  stream_a:
    documenter: claude-sonnet-4-5-20250929
    validator: claude-haiku-4-5-20251001
  stream_b:
    documenter: gpt-4o
    validator: gpt-4o-mini
  comparator: claude-opus-4-5-20251101

convergence:
  max_iterations: 5
  call_graph_match_threshold: 0.98
  documentation_similarity_threshold: 0.95

Beads:
  server: https://your-org.atlassian.net
  project: LEGACY_DOC
  rebuild_project: REBUILD
  
static_analysis:
  python:
    tool: pycg
    fallback: pyan3
    
output:
  documentation_path: ./output/documentation.json
  call_graph_path: ./output/call_graph.json
  rebuild_tickets_path: ./output/rebuild_tickets.json
```

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-06 | Initial architecture |
| 2.0 | 2026-01-06 | Added tiered model approach, refined dual-stream rationale |
