"""
TwinScribe Command Line Interface.

This module provides the CLI entry point for the dual-stream documentation system.
"""

import click
from rich.console import Console

from twinscribe.version import __version__

console = Console()


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
@click.pass_context
def document(
    ctx: click.Context,
    codebase_path: str,
    language: str,
    config: str | None,
    output: str,
    max_iterations: int,
) -> None:
    """Generate documentation for a codebase.

    CODEBASE_PATH is the path to the codebase to document.
    """
    verbose = ctx.obj.get("verbose", False)

    console.print(f"[bold blue]TwinScribe v{__version__}[/bold blue]")
    console.print(f"Documenting: [green]{codebase_path}[/green]")
    console.print(f"Language: {language}")
    console.print(f"Output: {output}")

    if verbose:
        console.print(f"Max iterations: {max_iterations}")
        if config:
            console.print(f"Config: {config}")

    # TODO: Implement actual documentation pipeline
    console.print("\n[yellow]Documentation pipeline not yet implemented.[/yellow]")
    console.print("See twinscribe-782 blocking tasks for implementation status.")


@main.command()
@click.argument("codebase_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="call_graph.json",
    help="Output file for call graph",
)
@click.pass_context
def analyze(ctx: click.Context, codebase_path: str, output: str) -> None:
    """Run static analysis to extract call graph.

    CODEBASE_PATH is the path to the codebase to analyze.
    """
    console.print(f"[bold blue]TwinScribe Static Analysis[/bold blue]")
    console.print(f"Analyzing: [green]{codebase_path}[/green]")
    console.print(f"Output: {output}")

    # TODO: Implement static analysis
    console.print("\n[yellow]Static analysis not yet implemented.[/yellow]")


@main.command()
def models() -> None:
    """List available LLM models and their tiers."""
    console.print("[bold blue]TwinScribe Model Configuration[/bold blue]\n")

    console.print("[bold]Generation Tier[/bold] (~$3/M tokens)")
    console.print("  - Stream A: anthropic/claude-sonnet-4-5-20250929")
    console.print("  - Stream B: openai/gpt-4o")

    console.print("\n[bold]Validation Tier[/bold] (~$0.20/M tokens)")
    console.print("  - Stream A: anthropic/claude-haiku-4-5-20251001")
    console.print("  - Stream B: openai/gpt-4o-mini")

    console.print("\n[bold]Arbitration Tier[/bold] (~$15/M tokens)")
    console.print("  - Comparator: anthropic/claude-opus-4-5-20251101")


if __name__ == "__main__":
    main()
