"""
TwinScribe Metrics Collection and Logging System.

This module provides comprehensive metrics collection, aggregation, and
structured logging for the TwinScribe documentation generation system.

Key Components:
    - MetricsCollector: Central metrics collection and aggregation
    - ComponentMetrics: Per-component metric tracking
    - PhaseMetrics: Duration tracking for processing phases
    - CostMetrics: LLM cost calculation and tracking
    - StructuredLogger: JSON-formatted logging with context

Tracked Metrics (from spec section 8.1):
    - convergence_rate: Percentage of components that converged
    - iterations_to_converge: Average/max iterations needed
    - call_graph_precision: Precision of call graph matches
    - call_graph_recall: Recall of call graph matches
    - beads_ticket_rate: Tickets created per component
    - cost_per_component: LLM cost breakdown by component
    - duration: Time tracking for each phase
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, Field, computed_field


class ProcessingPhase(str, Enum):
    """Processing phases for duration tracking."""

    INITIALIZATION = "initialization"
    PARSING = "parsing"
    STATIC_ANALYSIS = "static_analysis"
    STREAM_A_DOCUMENTATION = "stream_a_documentation"
    STREAM_B_DOCUMENTATION = "stream_b_documentation"
    STREAM_A_VALIDATION = "stream_a_validation"
    STREAM_B_VALIDATION = "stream_b_validation"
    COMPARISON = "comparison"
    CONVERGENCE_CHECK = "convergence_check"
    VERIFICATION = "verification"
    BEADS_TICKET_CREATION = "beads_ticket_creation"
    OUTPUT_GENERATION = "output_generation"
    TOTAL = "total"


class MetricCategory(str, Enum):
    """Categories for metric organization."""

    CONVERGENCE = "convergence"
    CALL_GRAPH = "call_graph"
    COST = "cost"
    DURATION = "duration"
    TICKETS = "tickets"
    QUALITY = "quality"


@dataclass
class PhaseMetrics:
    """Metrics for a single processing phase.

    Attributes:
        phase: Phase identifier
        start_time: When phase started
        end_time: When phase ended
        duration_seconds: Duration in seconds
        success: Whether phase completed successfully
        error_message: Error message if failed
        metadata: Additional phase-specific data
    """

    phase: ProcessingPhase
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    success: bool = True
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        """Mark phase as started."""
        self.start_time = datetime.utcnow()

    def stop(self, success: bool = True, error: str | None = None) -> None:
        """Mark phase as stopped.

        Args:
            success: Whether phase succeeded
            error: Error message if failed
        """
        self.end_time = datetime.utcnow()
        self.success = success
        self.error_message = error
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class CallGraphMetrics(BaseModel):
    """Metrics for call graph accuracy.

    Attributes:
        true_positives: Correctly identified calls
        false_positives: Incorrectly identified calls
        false_negatives: Missed calls
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        f1_score: Harmonic mean of precision and recall
    """

    true_positives: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    false_negatives: int = Field(default=0, ge=0)

    @computed_field
    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        total = self.true_positives + self.false_positives
        if total == 0:
            return 0.0
        return self.true_positives / total

    @computed_field
    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total

    @computed_field
    @property
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


class CostMetrics(BaseModel):
    """Cost tracking for LLM usage.

    Attributes:
        prompt_tokens: Total prompt tokens used
        completion_tokens: Total completion tokens used
        total_tokens: Total tokens used
        cost_usd: Total cost in USD
        requests_count: Number of API requests
        by_model: Breakdown by model
        by_role: Breakdown by role (documenter, validator, etc.)
    """

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    requests_count: int = Field(default=0, ge=0)
    by_model: dict[str, dict[str, Any]] = Field(default_factory=dict)
    by_role: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        model: str = "unknown",
        role: str = "unknown",
    ) -> None:
        """Add token usage to metrics.

        Args:
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            cost_usd: Cost in USD
            model: Model name
            role: Role (documenter, validator, comparator)
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.cost_usd += cost_usd
        self.requests_count += 1

        # Track by model
        if model not in self.by_model:
            self.by_model[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            }
        self.by_model[model]["prompt_tokens"] += prompt_tokens
        self.by_model[model]["completion_tokens"] += completion_tokens
        self.by_model[model]["cost_usd"] += cost_usd
        self.by_model[model]["requests"] += 1

        # Track by role
        if role not in self.by_role:
            self.by_role[role] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "requests": 0,
            }
        self.by_role[role]["prompt_tokens"] += prompt_tokens
        self.by_role[role]["completion_tokens"] += completion_tokens
        self.by_role[role]["cost_usd"] += cost_usd
        self.by_role[role]["requests"] += 1


class ConvergenceMetrics(BaseModel):
    """Metrics for convergence tracking.

    Attributes:
        total_components: Total number of components processed
        converged_count: Number that converged
        forced_count: Number that hit max iterations
        failed_count: Number that failed to converge
        iterations_to_converge: List of iterations needed per component
        convergence_rate: Percentage that converged
        average_iterations: Average iterations to convergence
        max_iterations: Maximum iterations used
    """

    total_components: int = Field(default=0, ge=0)
    converged_count: int = Field(default=0, ge=0)
    forced_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    iterations_to_converge: list[int] = Field(default_factory=list)

    @computed_field
    @property
    def convergence_rate(self) -> float:
        """Percentage of components that converged."""
        if self.total_components == 0:
            return 0.0
        return self.converged_count / self.total_components

    @computed_field
    @property
    def average_iterations(self) -> float:
        """Average iterations to converge."""
        if not self.iterations_to_converge:
            return 0.0
        return sum(self.iterations_to_converge) / len(self.iterations_to_converge)

    @computed_field
    @property
    def max_iterations(self) -> int:
        """Maximum iterations used."""
        if not self.iterations_to_converge:
            return 0
        return max(self.iterations_to_converge)

    def record_component(
        self,
        converged: bool,
        forced: bool,
        iterations: int,
    ) -> None:
        """Record component convergence result.

        Args:
            converged: Whether component converged
            forced: Whether max iterations was reached
            iterations: Iterations used
        """
        self.total_components += 1
        self.iterations_to_converge.append(iterations)

        if converged and not forced:
            self.converged_count += 1
        elif forced:
            self.forced_count += 1
        else:
            self.failed_count += 1


class BeadsTicketMetrics(BaseModel):
    """Metrics for Beads ticket creation.

    Attributes:
        total_tickets: Total tickets created
        tickets_by_type: Breakdown by ticket type
        tickets_by_severity: Breakdown by severity
        components_with_tickets: Set of components with tickets
        ticket_rate: Tickets per component
    """

    total_tickets: int = Field(default=0, ge=0)
    tickets_by_type: dict[str, int] = Field(default_factory=dict)
    tickets_by_severity: dict[str, int] = Field(default_factory=dict)
    components_with_tickets: set[str] = Field(default_factory=set)
    total_components: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def ticket_rate(self) -> float:
        """Tickets per component."""
        if self.total_components == 0:
            return 0.0
        return self.total_tickets / self.total_components

    def record_ticket(
        self,
        component_id: str,
        ticket_type: str,
        severity: str,
    ) -> None:
        """Record a ticket creation.

        Args:
            component_id: Component the ticket is for
            ticket_type: Type of ticket
            severity: Severity level
        """
        self.total_tickets += 1
        self.components_with_tickets.add(component_id)

        if ticket_type not in self.tickets_by_type:
            self.tickets_by_type[ticket_type] = 0
        self.tickets_by_type[ticket_type] += 1

        if severity not in self.tickets_by_severity:
            self.tickets_by_severity[severity] = 0
        self.tickets_by_severity[severity] += 1


@dataclass
class ComponentMetrics:
    """Metrics for a single component.

    Tracks all metrics related to processing a single code component.

    Attributes:
        component_id: Unique component identifier
        start_time: When processing started
        end_time: When processing ended
        iterations: Iterations to convergence
        converged: Whether component converged
        forced_convergence: Whether max iterations reached
        call_graph: Call graph accuracy metrics
        cost: Cost metrics for this component
        phase_durations: Duration of each phase
        verification_scores: Verification strategy scores
        tickets_created: Number of tickets created
        error: Error message if processing failed
    """

    component_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    iterations: int = 0
    converged: bool = False
    forced_convergence: bool = False
    call_graph: CallGraphMetrics = field(default_factory=CallGraphMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    phase_durations: dict[str, float] = field(default_factory=dict)
    verification_scores: dict[str, float] = field(default_factory=dict)
    tickets_created: int = 0
    error: str | None = None

    @property
    def total_duration_seconds(self) -> float:
        """Total processing duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def complete(
        self,
        converged: bool = False,
        forced: bool = False,
        error: str | None = None,
    ) -> None:
        """Mark component processing as complete.

        Args:
            converged: Whether convergence was achieved
            forced: Whether max iterations was reached
            error: Error message if failed
        """
        self.end_time = datetime.utcnow()
        self.converged = converged
        self.forced_convergence = forced
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_id": self.component_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "iterations": self.iterations,
            "converged": self.converged,
            "forced_convergence": self.forced_convergence,
            "call_graph": {
                "precision": self.call_graph.precision,
                "recall": self.call_graph.recall,
                "f1_score": self.call_graph.f1_score,
                "true_positives": self.call_graph.true_positives,
                "false_positives": self.call_graph.false_positives,
                "false_negatives": self.call_graph.false_negatives,
            },
            "cost": {
                "prompt_tokens": self.cost.prompt_tokens,
                "completion_tokens": self.cost.completion_tokens,
                "total_tokens": self.cost.total_tokens,
                "cost_usd": self.cost.cost_usd,
                "requests_count": self.cost.requests_count,
            },
            "phase_durations": self.phase_durations,
            "verification_scores": self.verification_scores,
            "tickets_created": self.tickets_created,
            "error": self.error,
        }


class MetricsCollector:
    """Central metrics collection and aggregation.

    Thread-safe collector for all TwinScribe metrics with support for
    concurrent component processing and phase tracking.

    Usage:
        collector = MetricsCollector()

        # Start a component
        collector.start_component("my_function")

        # Track phases
        async with collector.phase_context(ProcessingPhase.PARSING):
            # ... do parsing ...

        # Record call graph results
        collector.record_call_graph("my_function", tp=5, fp=1, fn=2)

        # Record costs
        collector.record_cost("my_function", prompt=100, completion=50, cost=0.01)

        # Complete component
        collector.complete_component("my_function", converged=True)

        # Export metrics
        summary = collector.get_summary()
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()
        self._start_time = datetime.utcnow()

        # Per-component metrics
        self._components: dict[str, ComponentMetrics] = {}

        # Aggregate metrics
        self._convergence = ConvergenceMetrics()
        self._cost = CostMetrics()
        self._tickets = BeadsTicketMetrics()

        # Phase tracking
        self._phases: dict[ProcessingPhase, PhaseMetrics] = {}
        self._active_phases: dict[ProcessingPhase, float] = {}

        # Run metadata
        self._run_id: str | None = None
        self._metadata: dict[str, Any] = {}

    def set_run_metadata(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set metadata for this metrics collection run.

        Args:
            run_id: Unique run identifier
            metadata: Additional metadata
        """
        self._run_id = run_id
        if metadata:
            self._metadata.update(metadata)

    def start_component(self, component_id: str) -> ComponentMetrics:
        """Start tracking a component.

        Args:
            component_id: Unique component identifier

        Returns:
            ComponentMetrics instance for the component
        """
        metrics = ComponentMetrics(component_id=component_id)
        self._components[component_id] = metrics
        return metrics

    def get_component(self, component_id: str) -> ComponentMetrics | None:
        """Get metrics for a component.

        Args:
            component_id: Component identifier

        Returns:
            ComponentMetrics or None if not found
        """
        return self._components.get(component_id)

    def complete_component(
        self,
        component_id: str,
        converged: bool = False,
        forced: bool = False,
        error: str | None = None,
    ) -> None:
        """Complete component processing.

        Args:
            component_id: Component identifier
            converged: Whether convergence achieved
            forced: Whether max iterations reached
            error: Error message if failed
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.complete(converged=converged, forced=forced, error=error)

            # Update convergence aggregates
            self._convergence.record_component(
                converged=converged and not forced,
                forced=forced,
                iterations=metrics.iterations,
            )

    def record_iteration(self, component_id: str, iteration: int) -> None:
        """Record an iteration for a component.

        Args:
            component_id: Component identifier
            iteration: Current iteration number
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.iterations = iteration

    def record_call_graph(
        self,
        component_id: str,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> None:
        """Record call graph metrics for a component.

        Args:
            component_id: Component identifier
            true_positives: Correctly identified calls
            false_positives: Incorrectly identified calls
            false_negatives: Missed calls
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.call_graph = CallGraphMetrics(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )

    def record_cost(
        self,
        component_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        model: str = "unknown",
        role: str = "unknown",
    ) -> None:
        """Record LLM cost for a component.

        Args:
            component_id: Component identifier
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            cost_usd: Cost in USD
            model: Model name
            role: Role (documenter, validator, etc.)
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.cost.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
                model=model,
                role=role,
            )

        # Also update global cost metrics
        self._cost.add_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            model=model,
            role=role,
        )

    def record_verification_scores(
        self,
        component_id: str,
        scores: dict[str, float],
    ) -> None:
        """Record verification scores for a component.

        Args:
            component_id: Component identifier
            scores: Dictionary of score names to values
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.verification_scores.update(scores)

    def record_ticket(
        self,
        component_id: str,
        ticket_type: str,
        severity: str,
    ) -> None:
        """Record a Beads ticket creation.

        Args:
            component_id: Component identifier
            ticket_type: Type of ticket
            severity: Severity level
        """
        self._tickets.record_ticket(
            component_id=component_id,
            ticket_type=ticket_type,
            severity=severity,
        )

        metrics = self._components.get(component_id)
        if metrics:
            metrics.tickets_created += 1

    def start_phase(self, phase: ProcessingPhase) -> None:
        """Start a processing phase.

        Args:
            phase: Phase to start
        """
        if phase not in self._phases:
            self._phases[phase] = PhaseMetrics(phase=phase)
        self._phases[phase].start()
        self._active_phases[phase] = time.time()

    def stop_phase(
        self,
        phase: ProcessingPhase,
        success: bool = True,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stop a processing phase.

        Args:
            phase: Phase to stop
            success: Whether phase succeeded
            error: Error message if failed
            metadata: Additional phase metadata
        """
        if phase in self._phases:
            self._phases[phase].stop(success=success, error=error)
            if metadata:
                self._phases[phase].metadata.update(metadata)
        if phase in self._active_phases:
            del self._active_phases[phase]

    @contextmanager
    def phase_context_sync(
        self,
        phase: ProcessingPhase,
        metadata: dict[str, Any] | None = None,
    ):
        """Synchronous context manager for phase tracking.

        Args:
            phase: Phase to track
            metadata: Additional metadata

        Yields:
            None
        """
        self.start_phase(phase)
        try:
            yield
            self.stop_phase(phase, success=True, metadata=metadata)
        except Exception as e:
            self.stop_phase(phase, success=False, error=str(e), metadata=metadata)
            raise

    @asynccontextmanager
    async def phase_context(
        self,
        phase: ProcessingPhase,
        metadata: dict[str, Any] | None = None,
    ):
        """Async context manager for phase tracking.

        Args:
            phase: Phase to track
            metadata: Additional metadata

        Yields:
            None
        """
        self.start_phase(phase)
        try:
            yield
            self.stop_phase(phase, success=True, metadata=metadata)
        except Exception as e:
            self.stop_phase(phase, success=False, error=str(e), metadata=metadata)
            raise

    def record_component_phase(
        self,
        component_id: str,
        phase: ProcessingPhase,
        duration_seconds: float,
    ) -> None:
        """Record phase duration for a specific component.

        Args:
            component_id: Component identifier
            phase: Processing phase
            duration_seconds: Duration in seconds
        """
        metrics = self._components.get(component_id)
        if metrics:
            metrics.phase_durations[phase.value] = duration_seconds

    def get_aggregate_call_graph_metrics(self) -> CallGraphMetrics:
        """Get aggregated call graph metrics across all components.

        Returns:
            Aggregated CallGraphMetrics
        """
        total_tp = sum(c.call_graph.true_positives for c in self._components.values())
        total_fp = sum(c.call_graph.false_positives for c in self._components.values())
        total_fn = sum(c.call_graph.false_negatives for c in self._components.values())

        return CallGraphMetrics(
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Dictionary with all metrics
        """
        end_time = datetime.utcnow()
        total_duration = (end_time - self._start_time).total_seconds()

        # Update ticket total_components count
        self._tickets.total_components = len(self._components)

        # Get aggregate call graph metrics
        agg_call_graph = self.get_aggregate_call_graph_metrics()

        return {
            "run_id": self._run_id,
            "metadata": self._metadata,
            "timing": {
                "start_time": self._start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration,
            },
            "convergence": {
                "total_components": self._convergence.total_components,
                "converged_count": self._convergence.converged_count,
                "forced_count": self._convergence.forced_count,
                "failed_count": self._convergence.failed_count,
                "convergence_rate": self._convergence.convergence_rate,
                "average_iterations": self._convergence.average_iterations,
                "max_iterations": self._convergence.max_iterations,
            },
            "call_graph": {
                "precision": agg_call_graph.precision,
                "recall": agg_call_graph.recall,
                "f1_score": agg_call_graph.f1_score,
                "true_positives": agg_call_graph.true_positives,
                "false_positives": agg_call_graph.false_positives,
                "false_negatives": agg_call_graph.false_negatives,
            },
            "cost": {
                "total_cost_usd": self._cost.cost_usd,
                "prompt_tokens": self._cost.prompt_tokens,
                "completion_tokens": self._cost.completion_tokens,
                "total_tokens": self._cost.total_tokens,
                "requests_count": self._cost.requests_count,
                "cost_per_component": (
                    self._cost.cost_usd / len(self._components) if self._components else 0.0
                ),
                "by_model": self._cost.by_model,
                "by_role": self._cost.by_role,
            },
            "tickets": {
                "total_tickets": self._tickets.total_tickets,
                "ticket_rate": self._tickets.ticket_rate,
                "by_type": self._tickets.tickets_by_type,
                "by_severity": self._tickets.tickets_by_severity,
                "components_with_tickets": len(self._tickets.components_with_tickets),
            },
            "phases": {phase.value: pm.to_dict() for phase, pm in self._phases.items()},
            "components": {cid: cm.to_dict() for cid, cm in self._components.items()},
        }

    def export_json(
        self,
        filepath: str | Path,
        indent: int = 2,
    ) -> None:
        """Export metrics to JSON file.

        Args:
            filepath: Output file path
            indent: JSON indentation
        """
        summary = self.get_summary()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=indent, default=str)

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._start_time = datetime.utcnow()
        self._components.clear()
        self._convergence = ConvergenceMetrics()
        self._cost = CostMetrics()
        self._tickets = BeadsTicketMetrics()
        self._phases.clear()
        self._active_phases.clear()
        self._run_id = None
        self._metadata.clear()


class StructuredLogger:
    """JSON-formatted structured logger.

    Provides consistent JSON logging with context, metrics integration,
    and correlation IDs.

    Usage:
        logger = StructuredLogger("twinscribe.orchestrator")
        logger.info("Processing started", component_id="func1", iteration=1)
        logger.error("Processing failed", error="Timeout", component_id="func1")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        output: TextIO | None = None,
        json_format: bool = True,
    ) -> None:
        """Initialize the structured logger.

        Args:
            name: Logger name
            level: Logging level
            output: Output stream (defaults to stderr)
            json_format: Whether to use JSON format
        """
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._json_format = json_format
        self._context: dict[str, Any] = {}
        self._correlation_id: str | None = None

        # Set up handler if needed
        if not self._logger.handlers:
            handler = logging.StreamHandler(output or sys.stderr)
            handler.setLevel(level)

            if json_format:
                handler.setFormatter(logging.Formatter("%(message)s"))
            else:
                handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )

            self._logger.addHandler(handler)

    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context fields.

        Args:
            **kwargs: Context fields to set
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context.clear()

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for log correlation.

        Args:
            correlation_id: Correlation identifier
        """
        self._correlation_id = correlation_id

    def _format_message(
        self,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Format log message.

        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields

        Returns:
            Formatted message string
        """
        if self._json_format:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                "logger": self._logger.name,
            }

            if self._correlation_id:
                record["correlation_id"] = self._correlation_id

            # Add context
            record.update(self._context)

            # Add extra fields
            record.update(kwargs)

            return json.dumps(record, default=str)
        else:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} {extra}".strip()

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._logger.debug(self._format_message("DEBUG", message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._logger.info(self._format_message("INFO", message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._logger.warning(self._format_message("WARNING", message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._logger.error(self._format_message("ERROR", message, **kwargs))

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message.

        Args:
            message: Log message
            **kwargs: Additional fields
        """
        self._logger.critical(self._format_message("CRITICAL", message, **kwargs))

    def log_metrics(
        self,
        metrics: dict[str, Any],
        category: MetricCategory,
        component_id: str | None = None,
    ) -> None:
        """Log metrics data.

        Args:
            metrics: Metrics dictionary
            category: Metric category
            component_id: Optional component ID
        """
        self.info(
            f"Metrics: {category.value}",
            category=category.value,
            component_id=component_id,
            metrics=metrics,
        )

    def log_phase_start(
        self,
        phase: ProcessingPhase,
        component_id: str | None = None,
    ) -> None:
        """Log phase start.

        Args:
            phase: Processing phase
            component_id: Optional component ID
        """
        self.info(
            f"Phase started: {phase.value}",
            phase=phase.value,
            component_id=component_id,
            event="phase_start",
        )

    def log_phase_end(
        self,
        phase: ProcessingPhase,
        duration_seconds: float,
        success: bool = True,
        component_id: str | None = None,
        error: str | None = None,
    ) -> None:
        """Log phase end.

        Args:
            phase: Processing phase
            duration_seconds: Phase duration
            success: Whether phase succeeded
            component_id: Optional component ID
            error: Error message if failed
        """
        level = "info" if success else "error"
        getattr(self, level)(
            f"Phase completed: {phase.value}",
            phase=phase.value,
            duration_seconds=duration_seconds,
            success=success,
            component_id=component_id,
            error=error,
            event="phase_end",
        )


# Module-level default instances
_default_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the default metrics collector.

    Returns:
        MetricsCollector singleton instance
    """
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


def get_structured_logger(name: str, json_format: bool = True) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name
        json_format: Whether to use JSON format

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, json_format=json_format)
