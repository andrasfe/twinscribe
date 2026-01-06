# Dual-Stream Code Documentation System - Architecture Documents

## Overview

This directory contains the detailed architecture design documents for the Dual-Stream Code Documentation System. These documents provide implementation guidance for the engineering team.

## Document Index

| Document | Description | Key Components |
|----------|-------------|----------------|
| [agent_architecture.md](agent_architecture.md) | Documenter and Validator agent design | `BaseAgent`, `DocumenterAgent`, `ValidatorAgent`, `DocumentationStream` |
| [comparator_architecture.md](comparator_architecture.md) | Comparator agent and arbitration logic | `ComparatorAgent`, `DiscrepancyDetector`, `ResolutionEngine`, `ConvergenceCriteria` |
| [orchestrator_architecture.md](orchestrator_architecture.md) | Orchestrator workflow and iteration loop | `DualStreamOrchestrator`, `ComponentDiscovery`, `DependencyGraphBuilder` |

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DualStreamOrchestrator                               │
│                                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ Component       │  │ Static Analysis  │  │ Beads Lifecycle             │ │
│  │ Discovery       │  │ Oracle           │  │ Manager                     │ │
│  └────────┬────────┘  └────────┬─────────┘  └─────────────┬───────────────┘ │
│           │                    │                          │                  │
└───────────┼────────────────────┼──────────────────────────┼──────────────────┘
            │                    │                          │
            ▼                    ▼                          │
┌───────────────────────────────────────────────────────────┼──────────────────┐
│                         ITERATION LOOP                    │                  │
│                                                           │                  │
│  ┌────────────────────┐    ┌────────────────────┐        │                  │
│  │    Stream A        │    │    Stream B        │        │                  │
│  │                    │    │                    │        │                  │
│  │  ┌──────────────┐  │    │  ┌──────────────┐  │        │                  │
│  │  │ Documenter   │  │    │  │ Documenter   │  │        │                  │
│  │  │ (Sonnet)     │  │    │  │ (GPT-4o)     │  │        │                  │
│  │  └──────┬───────┘  │    │  └──────┬───────┘  │        │                  │
│  │         │          │    │         │          │        │                  │
│  │         ▼          │    │         ▼          │        │                  │
│  │  ┌──────────────┐  │    │  ┌──────────────┐  │        │                  │
│  │  │ Validator    │  │    │  │ Validator    │  │        │                  │
│  │  │ (Haiku)      │  │    │  │ (GPT-4o-mini)│  │        │                  │
│  │  └──────────────┘  │    │  └──────────────┘  │        │                  │
│  └─────────┬──────────┘    └─────────┬──────────┘        │                  │
│            │                         │                   │                  │
│            └────────────┬────────────┘                   │                  │
│                         │                                │                  │
│                         ▼                                │                  │
│            ┌────────────────────────┐                    │                  │
│            │   Comparator Agent     │                    │                  │
│            │   (Claude Opus 4.5)    │◄───────────────────┘                  │
│            │                        │                                       │
│            │  - DiscrepancyDetector │                                       │
│            │  - ResolutionEngine    │                                       │
│            │  - ConvergenceCriteria │                                       │
│            └────────────────────────┘                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Agent Type Hierarchy

All agents inherit from `BaseAgent[InputT, OutputT]`, providing:
- Consistent error handling with retries
- Token usage tracking and cost estimation
- Metadata collection for audit trail
- Type-safe input/output contracts

### 2. Model Tiering

| Tier | Models | Cost | Purpose |
|------|--------|------|---------|
| Generation | Claude Sonnet 4.5, GPT-4o | ~$3/M | Documentation writing |
| Validation | Claude Haiku 4.5, GPT-4o-mini | ~$0.20/M | Verification checks |
| Arbitration | Claude Opus 4.5 | ~$15/M | Comparison and judgment |

### 3. Ground Truth Strategy

Static analysis (PyCG, pyan3) provides authoritative call graph:
- Call graph discrepancies resolved by ground truth automatically
- Documentation content discrepancies use LLM judgment
- Low-confidence decisions (<0.7) escalate to Beads

### 4. Convergence Criteria

```python
MAX_ITERATIONS = 5
CALL_GRAPH_MATCH_RATE = 0.98
DOCUMENTATION_SIMILARITY = 0.95
MAX_OPEN_DISCREPANCIES = 2
```

## Package Structure (Proposed)

```
twinscribe/
├── __init__.py
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent abstract class
│   ├── protocols.py         # AgentInput, AgentOutput protocols
│   ├── prompts.py           # System prompt templates
│   ├── exceptions.py        # Agent-related exceptions
│   ├── documenter/
│   │   ├── __init__.py
│   │   ├── agent.py         # DocumenterAgent implementations
│   │   └── models.py        # DocumenterInput, DocumenterOutput
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── agent.py         # ValidatorAgent implementations
│   │   └── models.py        # ValidatorInput, ValidatorOutput
│   └── comparator/
│       ├── __init__.py
│       ├── agent.py         # ComparatorAgent
│       ├── models.py        # Discrepancy, ComparisonResult
│       ├── detector.py      # DiscrepancyDetector
│       ├── resolver.py      # ResolutionEngine
│       └── convergence.py   # ConvergenceCriteria
├── streams/
│   ├── __init__.py
│   └── documentation_stream.py  # DocumentationStream
├── orchestrator/
│   ├── __init__.py
│   ├── orchestrator.py      # DualStreamOrchestrator
│   ├── models.py            # OrchestratorState, DocumentationPackage
│   ├── discovery.py         # ComponentDiscovery
│   ├── dependency.py        # DependencyGraphBuilder
│   └── exceptions.py        # Orchestrator exceptions
├── analysis/
│   ├── __init__.py
│   └── static_oracle.py     # StaticAnalysisOracle
├── beads/
│   ├── __init__.py
│   └── lifecycle.py         # BeadsLifecycleManager
├── llm/
│   ├── __init__.py
│   └── client.py            # LLMClient (OpenRouter wrapper)
├── config/
│   ├── __init__.py
│   └── models.py            # Configuration Pydantic models
└── cli/
    ├── __init__.py
    └── main.py              # CLI entry point
```

## Implementation Dependencies

The architecture documents define these cross-cutting dependencies:

1. **LLMClient**: Needed by all agents (design in separate task)
2. **StaticAnalysisOracle**: Needed by validators and comparator
3. **BeadsLifecycleManager**: Needed by orchestrator for ticket management
4. **Configuration System**: Needed by all components (design in separate task)

## Testing Considerations

### Unit Testing
- Mock LLMClient for deterministic agent testing
- Mock StaticAnalysisOracle for validation testing
- Test state machines with known inputs

### Integration Testing
- End-to-end flow with sample codebase
- Stream parallelization verification
- Convergence behavior validation

### Contract Testing
- JSON schema validation for all I/O models
- Protocol compliance verification

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-06 | Systems Architect | Initial architecture documents |
