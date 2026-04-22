# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import store
import ingest
import search as search_mod

app = typer.Typer(help="Atlas -- local AI search across your personal data.")
console = Console()


def _print_results(results: list):
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold", show_lines=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Path / Contact", style="dim")
    table.add_column("Snippet")
    table.add_column("Score", justify="right", style="green")
    for r in results:
        table.add_row(r.source_type, r.source_path, r.snippet, str(r.score))
    console.print(table)


@app.command()
def index(
    path: Optional[Path] = typer.Argument(None, help="Directory to index (filesystem)."),
    imessage: bool = typer.Option(False, "--imessage", help="Index iMessage history."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Index a directory and/or iMessage history."""
    if path is None and not imessage:
        console.print("[red]Provide a path to index or use --imessage.[/red]")
        raise typer.Exit(1)

    if path is not None:
        if not path.exists():
            console.print(f"[red]Path does not exist: {path}[/red]")
            raise typer.Exit(1)
        console.print(f"[bold]Indexing filesystem:[/bold] {path}")
        count = ingest.index_filesystem(path.expanduser().resolve(), verbose=verbose)
        console.print(f"[green]Indexed {count} files[/green]")

    if imessage:
        console.print("[bold]Indexing iMessage...[/bold]")
        try:
            count = ingest.index_imessage(verbose=verbose)
            console.print(f"[green]Indexed {count} conversations[/green]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)


@app.command()
def reindex(
    path: Path = typer.Argument(..., help="Directory to re-index from scratch."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Clear the index and rebuild it from a directory.

    Required after upgrades that change how embeddings are generated
    (e.g., adding path context to chunks).
    """
    console.print("[yellow]Clearing existing index...[/yellow]")
    store.clear()
    console.print(f"[bold]Re-indexing:[/bold] {path}")
    count = ingest.index_filesystem(path.expanduser().resolve(), verbose=verbose)
    console.print(f"[green]Re-indexed {count} files ({store.count()} chunks total)[/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query."),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results to return."),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source: filesystem or imessage."),
):
    """Semantic search across all indexed data."""
    results = search_mod.search(query, n_results=limit, source_filter=source)
    _print_results(results)


@app.command()
def ask(
    query: str = typer.Argument(..., help="Conversational query (parsed by local LLM)."),
    limit: int = typer.Option(10, "--limit", "-n"),
):
    """Conversational search: the LLM extracts intent and routes the query.

    Examples:
      python cli.py ask "please find the directory with my cs107 homework"
      python cli.py ask "find texts from Alex about the contract"
      python cli.py ask "what files do I have on machine learning"
    """
    console.print(f"[dim]Parsing query...[/dim]")
    results = search_mod.ask(query, n_results=limit)
    _print_results(results)


@app.command()
def status():
    """Show index statistics."""
    total = store.count()
    console.print(f"[bold]Atlas index[/bold]: {total} chunks stored")
    if total == 0:
        console.print("[yellow]Nothing indexed yet. Run: python cli.py index <path>[/yellow]")


if __name__ == "__main__":
    app()
