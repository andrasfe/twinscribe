"""
TwinScribe Command Line Interface.

This module provides the CLI entry point for the dual-stream documentation system.
"""

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from twinscribe.version import __version__

console = Console()


def run_async(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.run(coro)


@click.group()
@click.version_option(version=__version__, prog_name="twinscribe")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """TwinScribe: Dual-Stream Code Documentation System.

    Generate accurate code documentation with call graph linkages using
    a multi-agent architecture with static analysis validation.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("codebase_path", type=click.Path(exists=True))
@click.option(
    "--language",
    "-l",
    type=click.Choice(["python", "java", "javascript"]),
    default="python",
    help="Programming language of the codebase",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="output",
    help="Output directory for generated documentation",
)
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum convergence iterations",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Analyze without creating Beads tickets or modifying external systems",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Run both documentation streams in parallel (may hit rate limits)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output with detailed progress",
)
@click.option(
    "--delay",
    "-d",
    type=float,
    default=2.0,
    help="Delay in seconds between API calls to avoid rate limits (default: 2.0)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from the last checkpoint if one exists",
)
@click.option(
    "--resume-run-id",
    type=str,
    default=None,
    help="Resume a specific run by its ID (use with --resume)",
)
@click.pass_context
def document(
    ctx: click.Context,
    codebase_path: str,
    language: str,
    config: str | None,
    output: str,
    max_iterations: int,
    dry_run: bool,
    parallel: bool,
    verbose: bool,
    delay: float,
    resume: bool,
    resume_run_id: str | None,
) -> None:
    """Generate documentation for a codebase.

    CODEBASE_PATH is the path to the codebase to document.

    Use --resume to continue from the last checkpoint if an incomplete run exists.
    """
    # Allow verbose from either global or local flag
    verbose = verbose or ctx.obj.get("verbose", False)

    console.print(
        Panel(
            f"[bold blue]TwinScribe v{__version__}[/bold blue]\n"
            "Dual-Stream Code Documentation System",
            title="TwinScribe",
        )
    )

    # Display configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Codebase", str(Path(codebase_path).resolve()))
    config_table.add_row("Language", language)
    config_table.add_row("Output", output)
    config_table.add_row("Max Iterations", str(max_iterations))
    if dry_run:
        config_table.add_row("Mode", "[yellow]Dry Run[/yellow]")
    if resume:
        config_table.add_row("Mode", "[cyan]Resume[/cyan]")
    if config:
        config_table.add_row("Config File", config)
    console.print(config_table)
    console.print()

    # Handle resume logic
    checkpoint_state = None
    if resume:
        checkpoint_state = _handle_resume(output, codebase_path, resume_run_id, verbose)
        if checkpoint_state is None and not resume_run_id:
            # No resumable runs found, continue with fresh run
            console.print("[dim]No incomplete runs found, starting fresh.[/dim]")

    try:
        result = run_async(
            _run_document_pipeline(
                codebase_path=codebase_path,
                language=language,
                config_path=config,
                output_dir=output,
                max_iterations=max_iterations,
                dry_run=dry_run,
                parallel=parallel,
                verbose=verbose,
                delay=delay,
                checkpoint_state=checkpoint_state,
            )
        )

        if result:
            _display_documentation_summary(result, output)
        else:
            console.print("[yellow]Documentation pipeline completed with no results.[/yellow]")

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


def _handle_resume(
    output_dir: str,
    codebase_path: str,
    resume_run_id: str | None,
    verbose: bool,
) -> "CheckpointState | None":
    """Handle resume logic: find and load checkpoint state.

    Args:
        output_dir: Output directory containing checkpoints
        codebase_path: Path to the codebase being documented
        resume_run_id: Optional specific run ID to resume
        verbose: Whether to show verbose output

    Returns:
        CheckpointState if a resumable run is found, None otherwise
    """
    from twinscribe.orchestrator.checkpoint import CheckpointManager, CheckpointState

    checkpoint_dir = Path(output_dir) / "checkpoints"

    # Find resumable runs for this specific codebase
    resumable_runs = CheckpointManager.find_resumable_runs(
        checkpoint_dir, codebase_path=codebase_path
    )

    if not resumable_runs:
        if resume_run_id:
            console.print(f"[red]Error:[/red] No checkpoint found for run ID: {resume_run_id}")
            sys.exit(1)
        return None

    # Select run to resume
    selected_run = None
    if resume_run_id:
        # Find specific run
        selected_run = next(
            (r for r in resumable_runs if r["run_id"] == resume_run_id),
            None,
        )
        if not selected_run:
            console.print(f"[red]Error:[/red] Run ID '{resume_run_id}' not found.")
            console.print("[dim]Available runs:[/dim]")
            for run in resumable_runs[:5]:
                console.print(f"  - {run['run_id']}")
            sys.exit(1)
    else:
        # Use most recent run, but offer choice if multiple
        if len(resumable_runs) == 1:
            selected_run = resumable_runs[0]
        else:
            # Show available runs and pick most recent
            console.print("[bold]Found incomplete runs:[/bold]")
            resume_table = Table(show_header=True)
            resume_table.add_column("#", style="dim")
            resume_table.add_column("Run ID", style="cyan")
            resume_table.add_column("Started", style="green")
            resume_table.add_column("Components", style="yellow")
            resume_table.add_column("Iteration", style="magenta")

            for i, run in enumerate(resumable_runs[:5], 1):
                started = run.get("started_at", "")[:19]  # Truncate timestamp
                resume_table.add_row(
                    str(i),
                    run["run_id"],
                    started,
                    str(run["components_processed"]),
                    str(run["last_iteration"]),
                )

            console.print(resume_table)
            console.print()
            console.print("[dim]Resuming most recent run. Use --resume-run-id to select a specific run.[/dim]")
            selected_run = resumable_runs[0]

    # Load checkpoint and build state
    run_id = selected_run["run_id"]
    console.print(f"[cyan]Resuming run:[/cyan] {run_id}")
    console.print(
        f"[dim]  Components processed: {selected_run['components_processed']} "
        f"(A: {selected_run.get('stream_a_processed', '?')}, "
        f"B: {selected_run.get('stream_b_processed', '?')})[/dim]"
    )
    console.print(f"[dim]  Last iteration: {selected_run['last_iteration']}[/dim]")

    # Create checkpoint manager and build state
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        run_id=run_id,
    )
    checkpoint_state = CheckpointState.build_state(checkpoint_manager)

    return checkpoint_state


async def _run_document_pipeline(
    codebase_path: str,
    language: str,
    config_path: str | None,
    output_dir: str,
    max_iterations: int,
    dry_run: bool,
    parallel: bool,
    verbose: bool,
    delay: float,
    checkpoint_state: "CheckpointState | None" = None,
) -> dict | None:
    """Run the documentation pipeline asynchronously.

    Args:
        codebase_path: Path to the codebase to document
        language: Programming language
        config_path: Optional path to config file
        output_dir: Output directory
        max_iterations: Maximum iterations
        dry_run: Whether to run in dry-run mode
        parallel: Whether to run streams in parallel
        verbose: Whether to show verbose output
        delay: Delay between API calls
        checkpoint_state: Optional checkpoint state for resuming

    Returns:
        Dictionary with documentation results and metrics
    """
    import logging

    # Configure logging based on verbosity
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )
    # Set twinscribe loggers to appropriate level
    logging.getLogger("twinscribe").setLevel(log_level)

    from twinscribe.analysis.oracle import OracleFactory
    from twinscribe.config import (
        ConfigurationError,
        create_default_config,
        load_config,
        load_config_from_env,
    )
    from twinscribe.orchestrator.orchestrator import (
        DualStreamOrchestrator,
        OrchestratorConfig,
        OrchestratorPhase,
        OrchestratorState,
    )

    # Load configuration
    console.print("[dim]Loading configuration...[/dim]")
    try:
        if config_path:
            cfg = load_config(config_path)
        else:
            # Try to load from environment/default locations, fall back to creating default
            try:
                cfg = load_config_from_env()
            except FileNotFoundError:
                cfg = create_default_config(codebase_path)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise

    # Override config with CLI options
    # Note: We create a new config with overridden values
    from twinscribe.config.models import CodebaseConfig, Language, OutputConfig

    # Update codebase path and language
    CodebaseConfig(
        path=codebase_path,
        language=Language(language),
        exclude_patterns=cfg.codebase.exclude_patterns,
        include_patterns=cfg.codebase.include_patterns,
        entry_points=cfg.codebase.entry_points,
    )

    # Update output directory
    OutputConfig(
        base_dir=output_dir,
        documentation_file=cfg.output.documentation_file,
        call_graph_file=cfg.output.call_graph_file,
        rebuild_tickets_file=cfg.output.rebuild_tickets_file,
        convergence_report_file=cfg.output.convergence_report_file,
        metrics_file=cfg.output.metrics_file,
        create_dirs=True,
    )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize static analysis oracle
    console.print("[dim]Initializing static analysis...[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        oracle = OracleFactory.for_python(
            codebase_path=codebase_path,
            cache_enabled=cfg.static_analysis.cache_enabled,
        )
        await oracle.initialize()

        progress.update(task, description="Static analysis complete")

    # Display call graph summary
    if oracle.call_graph:
        console.print(
            f"[green]Call graph extracted:[/green] "
            f"{oracle.call_graph.edge_count} edges, "
            f"{oracle.call_graph.node_count} nodes"
        )

    # For now, we will run static analysis and save the call graph
    # The full orchestrator pipeline requires additional stream implementations
    # that may not be fully available yet

    # Save call graph output
    call_graph_path = output_path / cfg.output.call_graph_file
    if oracle.call_graph:
        call_graph_data = oracle.call_graph.model_dump(mode="json")
        with open(call_graph_path, "w") as f:
            json.dump(call_graph_data, f, indent=2)
        console.print(f"[green]Call graph saved to:[/green] {call_graph_path}")

    # Check if orchestrator dependencies are available
    try:
        from twinscribe.agents import (
            ComparatorConfig,
            ConcreteDocumentationStream,
            DocumenterConfig,
            StreamConfig,
            ValidatorConfig,
            create_comparator_agent,
        )
        from twinscribe.models.base import ModelTier, StreamId

        orchestrator_available = True
    except ImportError as e:
        orchestrator_available = False
        if verbose:
            console.print(f"[yellow]Note:[/yellow] Full orchestrator not available: {e}")

    if orchestrator_available:
        console.print("\n[bold]Running Dual-Stream Documentation Pipeline[/bold]")

        # Create orchestrator configuration with fail-fast behavior
        orch_config = OrchestratorConfig(
            max_iterations=max_iterations,
            parallel_components=10,
            parallel_streams=parallel,
            wait_for_beads=not dry_run,
            beads_timeout_hours=cfg.convergence.beads_ticket_timeout_hours,
            skip_validation=False,
            dry_run=dry_run,
            continue_on_error=False,  # Fail-fast: quit on first error
        )

        # Progress callback for live updates
        last_phase = [None]  # Use list to allow mutation in closure

        def on_progress(state: OrchestratorState):
            phase_names = {
                OrchestratorPhase.NOT_STARTED: "Not Started",
                OrchestratorPhase.INITIALIZING: "Initializing",
                OrchestratorPhase.DISCOVERING: "Discovering Components",
                OrchestratorPhase.DOCUMENTING: "Documenting",
                OrchestratorPhase.COMPARING: "Comparing Outputs",
                OrchestratorPhase.RESOLVING: "Resolving Discrepancies",
                OrchestratorPhase.FINALIZING: "Finalizing",
                OrchestratorPhase.COMPLETED: "Completed",
                OrchestratorPhase.FAILED: "Failed",
            }
            phase_name = phase_names.get(state.phase, state.phase.value)

            # Always show phase changes
            if state.phase != last_phase[0]:
                last_phase[0] = state.phase
                if state.phase == OrchestratorPhase.DISCOVERING:
                    console.print(f"[cyan]→ {phase_name}...[/cyan]")
                elif state.phase == OrchestratorPhase.DOCUMENTING:
                    console.print(f"[cyan]→ {phase_name} ({state.total_components} components)...[/cyan]")
                elif state.phase == OrchestratorPhase.COMPARING:
                    console.print(f"[cyan]→ {phase_name}...[/cyan]")
                elif state.phase == OrchestratorPhase.RESOLVING:
                    console.print(f"[cyan]→ {phase_name} ({state.pending_discrepancies} discrepancies)...[/cyan]")
                elif state.phase == OrchestratorPhase.FINALIZING:
                    console.print(f"[cyan]→ {phase_name}...[/cyan]")
                elif state.phase == OrchestratorPhase.COMPLETED:
                    console.print(f"[green]✓ {phase_name}[/green]")
                elif state.phase == OrchestratorPhase.FAILED:
                    console.print(f"[red]✗ {phase_name}[/red]")

            # Show progress during documenting phase
            if verbose and state.phase == OrchestratorPhase.DOCUMENTING:
                console.print(
                    f"[dim]  Iteration {state.iteration} | "
                    f"Processed: {state.processed_components}/{state.total_components} | "
                    f"Converged: {state.converged_components}[/dim]"
                )

        # Helper to detect provider from model name
        def get_provider(model_name: str) -> str:
            if "claude" in model_name.lower():
                return "anthropic"
            elif "gpt" in model_name.lower() or "o1" in model_name.lower():
                return "openai"
            return "openrouter"  # Default to openrouter

        # Create stream configurations using models config with fail-fast behavior
        stream_a_config = StreamConfig(
            stream_id=StreamId.STREAM_A,
            documenter_config=DocumenterConfig(
                agent_id="A1",
                stream_id=StreamId.STREAM_A,
                model_tier=ModelTier.GENERATION,
                provider=get_provider(cfg.models.stream_a.documenter),
                model_name=cfg.models.stream_a.documenter,
                max_tokens=4096,
                temperature=0.0,
            ),
            validator_config=ValidatorConfig(
                agent_id="A2",
                stream_id=StreamId.STREAM_A,
                model_tier=ModelTier.VALIDATION,
                provider=get_provider(cfg.models.stream_a.validator),
                model_name=cfg.models.stream_a.validator,
                max_tokens=2048,
                temperature=0.0,
            ),
            batch_size=5,
            max_retries=3,
            continue_on_error=False,  # Fail-fast: quit on first error
            rate_limit_delay=delay,
        )

        stream_b_config = StreamConfig(
            stream_id=StreamId.STREAM_B,
            documenter_config=DocumenterConfig(
                agent_id="B1",
                stream_id=StreamId.STREAM_B,
                model_tier=ModelTier.GENERATION,
                provider=get_provider(cfg.models.stream_b.documenter),
                model_name=cfg.models.stream_b.documenter,
                max_tokens=4096,
                temperature=0.0,
            ),
            validator_config=ValidatorConfig(
                agent_id="B2",
                stream_id=StreamId.STREAM_B,
                model_tier=ModelTier.VALIDATION,
                provider=get_provider(cfg.models.stream_b.validator),
                model_name=cfg.models.stream_b.validator,
                max_tokens=2048,
                temperature=0.0,
            ),
            batch_size=5,
            max_retries=3,
            continue_on_error=False,  # Fail-fast: quit on first error
            rate_limit_delay=delay,
        )

        # Create checkpoint manager for state persistence and resume capability
        from twinscribe.orchestrator.checkpoint import CheckpointManager

        # If resuming, use the run_id from checkpoint state
        if checkpoint_state is not None:
            checkpoint_manager = CheckpointManager(
                output_dir=output_dir,
                run_id=checkpoint_state.run_id,
            )
            console.print(f"[dim]Resuming run: {checkpoint_state.run_id}[/dim]")
        else:
            checkpoint_manager = CheckpointManager(output_dir=output_dir)
            checkpoint_manager.record_run_start(config={
                "codebase_path": codebase_path,
                "language": language,
                "max_iterations": max_iterations,
                "parallel": parallel,
                "dry_run": dry_run,
            })

        # Create streams with checkpoint manager
        stream_a = ConcreteDocumentationStream(stream_a_config, checkpoint_manager=checkpoint_manager)
        stream_b = ConcreteDocumentationStream(stream_b_config, checkpoint_manager=checkpoint_manager)

        # Create comparator
        comparator_config = ComparatorConfig(
            agent_id="C",
            model_tier=ModelTier.ARBITRATION,
            provider=get_provider(cfg.models.comparator),
            model_name=cfg.models.comparator,
            max_tokens=4096,
            temperature=0.0,
        )
        comparator = create_comparator_agent(config=comparator_config)

        # Create Beads manager if enabled and not dry run
        beads_manager = None
        if cfg.beads.enabled and not dry_run:
            try:
                from twinscribe.beads import BeadsLifecycleManager

                beads_manager = BeadsLifecycleManager()
            except ImportError:
                if verbose:
                    console.print("[yellow]Beads integration not available[/yellow]")

        # Create orchestrator with checkpoint manager for fail-fast error handling
        orchestrator = DualStreamOrchestrator(
            config=orch_config,
            static_oracle=oracle,
            stream_a=stream_a,
            stream_b=stream_b,
            comparator=comparator,
            beads_manager=beads_manager,
            checkpoint_manager=checkpoint_manager,
        )

        # Register progress callback
        orchestrator.on_progress(on_progress)

        # Run the pipeline with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task_desc = "Resuming documentation..." if checkpoint_state else "Documenting codebase..."
            task = progress.add_task(task_desc, total=max_iterations)

            try:
                # Use resume_from_checkpoint if we have checkpoint state
                if checkpoint_state is not None:
                    package = await orchestrator.resume_from_checkpoint(checkpoint_state)
                else:
                    package = await orchestrator.run()

                # Save documentation output
                doc_path = output_path / cfg.output.documentation_file
                doc_data = {
                    comp_id: doc.model_dump(mode="json")
                    for comp_id, doc in package.documentation.items()
                }
                with open(doc_path, "w") as f:
                    json.dump(doc_data, f, indent=2)
                console.print(f"[green]Documentation saved to:[/green] {doc_path}")

                # Save convergence report
                report_path = output_path / cfg.output.convergence_report_file
                report_data = package.convergence_report.model_dump(mode="json")
                with open(report_path, "w") as f:
                    json.dump(report_data, f, indent=2)
                console.print(f"[green]Convergence report saved to:[/green] {report_path}")

                # Save metrics
                metrics_path = output_path / cfg.output.metrics_file
                metrics_data = package.metrics.model_dump(mode="json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics_data, f, indent=2)
                console.print(f"[green]Metrics saved to:[/green] {metrics_path}")

                # Return summary
                return {
                    "components_documented": package.component_count,
                    "call_graph_edges": package.edge_count,
                    "converged": package.convergence_report.is_successful,
                    "iterations": package.convergence_report.total_iterations,
                    "metrics": package.metrics.model_dump(mode="json"),
                }

            except Exception as e:
                # Error state is already saved to checkpoint by orchestrator/stream
                # Show clear error message with checkpoint location for resume
                console.print(f"\n[red]Pipeline error:[/red] {e}")
                console.print(
                    f"[yellow]Checkpoint saved to:[/yellow] {checkpoint_manager.checkpoint_path}"
                )
                console.print(
                    "[dim]You can resume from this checkpoint once the error is resolved.[/dim]"
                )
                if verbose:
                    console.print_exception()
                # Re-raise to indicate failure (fail-fast behavior)
                raise

    else:
        # Return just the static analysis results
        return {
            "call_graph_edges": oracle.call_graph.edge_count if oracle.call_graph else 0,
            "call_graph_nodes": oracle.call_graph.node_count if oracle.call_graph else 0,
            "output_dir": str(output_path),
        }


def _display_documentation_summary(result: dict, output_dir: str) -> None:
    """Display a summary of the documentation results."""
    console.print()
    console.print(Panel("[bold green]Documentation Complete[/bold green]"))

    summary_table = Table(title="Summary Statistics", show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    if "components_documented" in result:
        summary_table.add_row("Components Documented", str(result["components_documented"]))
    if "call_graph_edges" in result:
        summary_table.add_row("Call Graph Edges", str(result["call_graph_edges"]))
    if "call_graph_nodes" in result:
        summary_table.add_row("Call Graph Nodes", str(result["call_graph_nodes"]))
    if "converged" in result:
        status = "[green]Yes[/green]" if result["converged"] else "[yellow]No[/yellow]"
        summary_table.add_row("Converged", status)
    if "iterations" in result:
        summary_table.add_row("Iterations", str(result["iterations"]))
    if "error" in result:
        summary_table.add_row("Status", f"[red]{result['error']}[/red]")

    summary_table.add_row("Output Directory", output_dir)

    console.print(summary_table)

    # Display metrics if available
    if "metrics" in result and result["metrics"]:
        metrics = result["metrics"]
        console.print()
        metrics_table = Table(title="Run Metrics", show_header=False, box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        if "duration_seconds" in metrics and metrics["duration_seconds"]:
            duration = metrics["duration_seconds"]
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            metrics_table.add_row("Duration", f"{minutes}m {seconds}s")
        if "cost" in metrics:
            cost = metrics["cost"]
            if isinstance(cost, dict) and "total" in cost:
                metrics_table.add_row("Total Cost", f"${cost['total']:.4f} USD")
        if "tokens_total" in metrics:
            metrics_table.add_row("Total Tokens", f"{metrics['tokens_total']:,}")

        console.print(metrics_table)


@main.command()
@click.argument("codebase_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="call_graph.json",
    help="Output file for call graph",
)
@click.option(
    "--language",
    "-l",
    type=click.Choice(["python", "java", "javascript"]),
    default="python",
    help="Programming language of the codebase",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "dot", "summary"]),
    default="json",
    help="Output format",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    codebase_path: str,
    output: str,
    language: str,
    output_format: str,
) -> None:
    """Run static analysis to extract call graph.

    CODEBASE_PATH is the path to the codebase to analyze.
    """
    verbose = ctx.obj.get("verbose", False)

    console.print(
        Panel(
            f"[bold blue]TwinScribe Static Analysis[/bold blue]\n"
            f"Extracting call graph from {Path(codebase_path).name}",
            title="Static Analysis",
        )
    )

    try:
        result = run_async(
            _run_analysis(
                codebase_path=codebase_path,
                output_file=output,
                language=language,
                output_format=output_format,
                verbose=verbose,
            )
        )

        _display_analysis_summary(result)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


async def _run_analysis(
    codebase_path: str,
    output_file: str,
    language: str,
    output_format: str,
    verbose: bool,
) -> dict:
    """Run static analysis asynchronously."""
    from twinscribe.analysis.oracle import OracleFactory

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        # Create and initialize oracle based on language
        if language == "python":
            oracle = OracleFactory.for_python(codebase_path)
        else:
            # For now, fall back to Python analyzer for unsupported languages
            console.print(
                f"[yellow]Warning:[/yellow] {language} analysis not fully supported, using Python analyzer"
            )
            oracle = OracleFactory.for_python(codebase_path)

        await oracle.initialize()

        progress.update(task, description="Analysis complete")

    call_graph = oracle.call_graph
    if not call_graph:
        raise RuntimeError("Failed to extract call graph")

    # Generate output based on format
    output_path = Path(output_file)

    if output_format == "json":
        data = call_graph.model_dump(mode="json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Call graph saved to:[/green] {output_path}")

    elif output_format == "dot":
        # Generate DOT format for GraphViz
        dot_content = _generate_dot(call_graph)
        with open(output_path, "w") as f:
            f.write(dot_content)
        console.print(f"[green]DOT graph saved to:[/green] {output_path}")

    elif output_format == "summary":
        # Just print summary, don't save file
        pass

    return {
        "edge_count": call_graph.edge_count,
        "node_count": call_graph.node_count,
        "source": call_graph.source,
        "output_file": str(output_path) if output_format != "summary" else None,
    }


def _generate_dot(call_graph) -> str:
    """Generate DOT format representation of call graph."""
    lines = ["digraph CallGraph {"]
    lines.append("    rankdir=LR;")
    lines.append("    node [shape=box];")
    lines.append("")

    # Add edges
    for edge in call_graph.edges:
        # Escape node names for DOT format
        caller = edge.caller.replace('"', '\\"')
        callee = edge.callee.replace('"', '\\"')
        lines.append(f'    "{caller}" -> "{callee}";')

    lines.append("}")
    return "\n".join(lines)


def _display_analysis_summary(result: dict) -> None:
    """Display analysis results summary."""
    console.print()
    summary_table = Table(title="Analysis Results", show_header=False, box=None)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total Edges", str(result["edge_count"]))
    summary_table.add_row("Total Nodes", str(result["node_count"]))
    summary_table.add_row("Analyzer", result["source"])
    if result.get("output_file"):
        summary_table.add_row("Output File", result["output_file"])

    console.print(summary_table)


@main.command()
def models() -> None:
    """List configured LLM models and their tiers."""
    from twinscribe.config import ensure_dotenv_loaded
    from twinscribe.config.models import ModelsConfig

    # Ensure .env is loaded
    ensure_dotenv_loaded()

    # Get the configured models (reads from environment)
    models_config = ModelsConfig()

    console.print(
        Panel(
            "[bold blue]TwinScribe Model Configuration[/bold blue]\n"
            "[dim]Models are configured via .env or environment variables[/dim]",
            title="Models",
        )
    )

    # Helper to detect provider from model name
    def get_provider(model_name: str) -> str:
        if "claude" in model_name.lower():
            return "Anthropic"
        elif "gpt" in model_name.lower() or "o1" in model_name.lower():
            return "OpenAI"
        elif "gemini" in model_name.lower():
            return "Google"
        elif "deepseek" in model_name.lower():
            return "DeepSeek"
        elif "grok" in model_name.lower():
            return "xAI"
        elif "glm" in model_name.lower():
            return "Zhipu AI"
        elif "minimax" in model_name.lower():
            return "MiniMax"
        return "OpenRouter"

    # Generation tier (Documenters)
    gen_table = Table(title="Generation Tier (Documenters)", show_header=True)
    gen_table.add_column("Stream", style="cyan")
    gen_table.add_column("Model", style="green")
    gen_table.add_column("Provider", style="dim")
    gen_table.add_row(
        "Stream A",
        models_config.stream_a.documenter,
        get_provider(models_config.stream_a.documenter),
    )
    gen_table.add_row(
        "Stream B",
        models_config.stream_b.documenter,
        get_provider(models_config.stream_b.documenter),
    )
    console.print(gen_table)
    console.print()

    # Validation tier
    val_table = Table(title="Validation Tier (Validators)", show_header=True)
    val_table.add_column("Stream", style="cyan")
    val_table.add_column("Model", style="green")
    val_table.add_column("Provider", style="dim")
    val_table.add_row(
        "Stream A",
        models_config.stream_a.validator,
        get_provider(models_config.stream_a.validator),
    )
    val_table.add_row(
        "Stream B",
        models_config.stream_b.validator,
        get_provider(models_config.stream_b.validator),
    )
    console.print(val_table)
    console.print()

    # Arbitration tier
    arb_table = Table(title="Arbitration Tier (Comparator)", show_header=True)
    arb_table.add_column("Role", style="cyan")
    arb_table.add_column("Model", style="green")
    arb_table.add_column("Provider", style="dim")
    arb_table.add_row(
        "Comparator",
        models_config.comparator,
        get_provider(models_config.comparator),
    )
    console.print(arb_table)

    # Show environment variables hint
    console.print()
    console.print("[dim]Configure models via environment variables:[/dim]")
    console.print("[dim]  STREAM_A_DOCUMENTER_MODEL, STREAM_A_VALIDATOR_MODEL[/dim]")
    console.print("[dim]  STREAM_B_DOCUMENTER_MODEL, STREAM_B_VALIDATOR_MODEL[/dim]")
    console.print("[dim]  COMPARATOR_MODEL[/dim]")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
def config(config: str | None) -> None:
    """Display current configuration."""
    from twinscribe.config import (
        ConfigurationError,
        load_config,
        load_config_from_env,
    )

    try:
        if config:
            cfg = load_config(config)
        else:
            cfg = load_config_from_env()

        console.print(
            Panel(
                "[bold blue]TwinScribe Configuration[/bold blue]",
                title="Configuration",
            )
        )

        # Codebase config
        console.print("[bold]Codebase[/bold]")
        console.print(f"  Path: {cfg.codebase.path}")
        console.print(f"  Language: {cfg.codebase.language.value}")
        console.print()

        # Models config
        console.print("[bold]Models[/bold]")
        console.print(f"  Stream A Documenter: {cfg.models.stream_a.documenter}")
        console.print(f"  Stream A Validator: {cfg.models.stream_a.validator}")
        console.print(f"  Stream B Documenter: {cfg.models.stream_b.documenter}")
        console.print(f"  Stream B Validator: {cfg.models.stream_b.validator}")
        console.print(f"  Comparator: {cfg.models.comparator}")
        console.print()

        # Convergence config
        console.print("[bold]Convergence[/bold]")
        console.print(f"  Max Iterations: {cfg.convergence.max_iterations}")
        console.print(f"  Call Graph Threshold: {cfg.convergence.call_graph_match_threshold}")
        console.print(
            f"  Doc Similarity Threshold: {cfg.convergence.documentation_similarity_threshold}"
        )
        console.print()

        # Output config
        console.print("[bold]Output[/bold]")
        console.print(f"  Base Directory: {cfg.output.base_dir}")
        console.print()

        # Beads config
        console.print("[bold]Beads Integration[/bold]")
        console.print(f"  Enabled: {cfg.beads.enabled}")
        if cfg.beads.enabled:
            console.print(f"  Project: {cfg.beads.project}")
            console.print(f"  Auto-create Tickets: {cfg.beads.auto_create_tickets}")

    except FileNotFoundError:
        console.print("[yellow]No configuration file found.[/yellow]")
        console.print("Create config.yaml or set TWINSCRIBE_CONFIG environment variable.")
        console.print()
        console.print("[dim]Default configuration will be used with codebase path from CLI.[/dim]")
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
