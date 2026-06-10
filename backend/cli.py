# -*- coding: utf-8 -*-
"""Atlas CLI — thin HTTP client over the local daemon.

The daemon is auto-started on first use and persists across CLI invocations,
keeping the embedding model, LLM, and vector cache permanently warm.
"""
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from config import (
    DAEMON_URL,
    DAEMON_PID_FILE,
    DAEMON_LOG_FILE,
    KEYCHAIN_SERVICE,
    KEYCHAIN_USERNAME,
    load_user_config,
    save_user_config,
)

app = typer.Typer(help="Atlas -- local AI search across your personal data.")
daemon_app = typer.Typer(help="Manage the Atlas daemon.")
config_app = typer.Typer(help="View and edit Atlas user config.")
app.add_typer(daemon_app, name="daemon")
app.add_typer(config_app, name="config")
console = Console()


# ---------------------------------------------------------------------------
# Daemon lifecycle helpers
# ---------------------------------------------------------------------------

def _daemon_pid() -> Optional[int]:
    if not DAEMON_PID_FILE.exists():
        return None
    try:
        pid = int(DAEMON_PID_FILE.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)  # signal 0 = check existence without sending
        return pid
    except ProcessLookupError:
        DAEMON_PID_FILE.unlink(missing_ok=True)
        return None


def _daemon_alive() -> bool:
    """Check if the daemon is responsive. Hits `/healthz` (no store/model
    state) so a bug in a deeper handler doesn't make the daemon look dead.
    A generous timeout covers the case where it's mid-request and slow to
    answer the health check."""
    try:
        r = requests.get(f"{DAEMON_URL}/healthz", timeout=5.0)
        return r.ok
    except requests.RequestException:
        return False


def _spawn_daemon() -> int:
    # Re-check inside the spawn path — the alive check could have raced with
    # a concurrently-starting daemon. Don't double-spawn.
    if _daemon_alive():
        return _daemon_pid() or 0
    DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    log_fp = DAEMON_LOG_FILE.open("a")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "main.py")],
        stdout=log_fp,
        stderr=log_fp,
        start_new_session=True,
    )
    DAEMON_PID_FILE.write_text(str(proc.pid))
    return proc.pid


def _wait_for_daemon(timeout: float = 300.0, log_start_pos: Optional[int] = None) -> bool:
    """Block until the daemon answers /status. If `log_start_pos` is given,
    tail the daemon log from that byte offset and surface lines from the
    `atlas.daemon` logger as a live spinner status, so the user sees which
    warmup stage we're in instead of a blank wait."""
    deadline = time.time() + timeout
    if log_start_pos is None:
        while time.time() < deadline:
            if _daemon_alive():
                return True
            time.sleep(0.3)
        return False

    log_pos = log_start_pos
    with console.status("[dim]Starting daemon...[/dim]") as status:
        while time.time() < deadline:
            if _daemon_alive():
                return True
            if DAEMON_LOG_FILE.exists():
                with DAEMON_LOG_FILE.open() as f:
                    f.seek(log_pos)
                    new = f.read()
                    log_pos = f.tell()
                for line in new.splitlines():
                    if "atlas.daemon:" in line:
                        msg = line.split("atlas.daemon:", 1)[-1].strip()
                        status.update(f"[dim]{msg}[/dim]")
            time.sleep(0.3)
    return False


def _log_size() -> int:
    return DAEMON_LOG_FILE.stat().st_size if DAEMON_LOG_FILE.exists() else 0


def _request_with_log_status(method: str, url: str, label: str, **kwargs) -> requests.Response:
    """Run an HTTP request in a background thread while tailing the daemon log
    and surfacing `atlas.daemon` lines as a live spinner status — same pattern
    as `_wait_for_daemon`, so the user sees stages instead of a blank wait."""
    result: dict = {}

    def _do():
        try:
            result["response"] = requests.request(method, url, **kwargs)
        except Exception as e:
            result["error"] = e

    log_pos = _log_size()
    t = threading.Thread(target=_do, daemon=True)
    t.start()
    with console.status(f"[dim]{label}[/dim]") as status:
        while t.is_alive():
            if DAEMON_LOG_FILE.exists():
                with DAEMON_LOG_FILE.open() as f:
                    f.seek(log_pos)
                    new = f.read()
                    log_pos = f.tell()
                for line in new.splitlines():
                    if "atlas.daemon:" in line:
                        msg = line.split("atlas.daemon:", 1)[-1].strip()
                        status.update(f"[dim]{msg}[/dim]")
            time.sleep(0.1)
    t.join()
    if "error" in result:
        raise result["error"]
    return result["response"]


def _ensure_daemon():
    """Auto-start the daemon if it's not already running."""
    if _daemon_alive():
        return
    log_pos = _log_size()
    _spawn_daemon()
    if not _wait_for_daemon(log_start_pos=log_pos):
        console.print(f"[red]Daemon failed to start. Check {DAEMON_LOG_FILE}[/red]")
        raise typer.Exit(1)
    console.print("[green]Daemon ready.[/green]")


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def _print_results(results: list[dict]):
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold", show_lines=True)
    table.add_column("Source", style="cyan", no_wrap=True)
    table.add_column("Path / Contact", style="dim")
    table.add_column("Snippet")
    table.add_column("Score", justify="right", style="green")
    for r in results:
        table.add_row(r["source_type"], r["source_path"], r["snippet"], str(r["score"]))
    console.print(table)


# ---------------------------------------------------------------------------
# User-facing commands
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query."),
    limit: int = typer.Option(10, "--limit", "-n"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter by source: filesystem, imessage, gcal, or gmail."),
):
    """Semantic search across all indexed data."""
    _ensure_daemon()
    r = _request_with_log_status(
        "GET",
        f"{DAEMON_URL}/search",
        label="Searching...",
        params={"q": query, "limit": limit, "source": source},
    )
    r.raise_for_status()
    _print_results(r.json())


@app.command()
def ask(
    query: str = typer.Argument(..., help="Conversational query (parsed by local LLM)."),
    limit: int = typer.Option(10, "--limit", "-n"),
):
    """Conversational search routed by the local LLM."""
    _ensure_daemon()
    r = _request_with_log_status(
        "POST",
        f"{DAEMON_URL}/ask",
        label="Thinking...",
        json={"query": query, "limit": limit},
    )
    r.raise_for_status()
    _print_results(r.json())


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer using retrieved chunks (RAG)."),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of source chunks to retrieve."),
):
    """RAG: retrieve top chunks, then synthesize an answer via the LLM with
    [1]/[2]-style citations. Slower than /search and /ask — also pays one
    LLM generation pass at the end."""
    _ensure_daemon()
    r = _request_with_log_status(
        "POST",
        f"{DAEMON_URL}/query",
        label="Retrieving + synthesizing...",
        json={"question": question, "limit": limit},
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    console.print()
    console.print(f"[bold]Answer:[/bold]\n{data['answer']}")
    console.print()
    console.print("[bold]Sources:[/bold]")
    _print_results(data["sources"])


def _poll_index_progress(label: str):
    """Poll /index/status until the job finishes, rendering a live progress bar."""
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(label, total=None)
        while True:
            time.sleep(0.5)
            r = requests.get(f"{DAEMON_URL}/index/status")
            r.raise_for_status()
            s = r.json()
            total = s.get("total") or None
            if total:
                progress.update(task, completed=s["indexed"], total=total)
            else:
                progress.update(task, completed=s["indexed"])
            if s["state"] == "done":
                progress.update(task, completed=s["indexed"], total=s["indexed"] or 1)
                return s
            if s["state"] == "error":
                progress.stop()
                console.print(f"[red]Index failed: {s['error']}[/red]")
                raise typer.Exit(1)


@app.command()
def index(
    path: Optional[Path] = typer.Argument(None, help="Directory to index (filesystem)."),
    imessage: bool = typer.Option(False, "--imessage", help="Index iMessage history."),
    gcal: bool = typer.Option(False, "--gcal", help="Index Google Calendar events."),
    gmail: bool = typer.Option(False, "--gmail", help="Index Gmail messages (opt-in; reads mail locally for search)."),
):
    """Index a directory, iMessage history, Google Calendar, and/or Gmail."""
    if path is None and not imessage and not gcal and not gmail:
        console.print("[red]Provide a path to index or use --imessage / --gcal / --gmail.[/red]")
        raise typer.Exit(1)
    _ensure_daemon()
    if path is not None:
        if not path.exists():
            console.print(f"[red]Path does not exist: {path}[/red]")
            raise typer.Exit(1)
        resolved = str(path.expanduser().resolve())
        r = requests.post(f"{DAEMON_URL}/index/filesystem", json={"path": resolved})
        if r.status_code == 409:
            console.print(f"[red]{r.json().get('detail')}[/red]")
            raise typer.Exit(1)
        r.raise_for_status()
        result = _poll_index_progress(f"Indexing {path.name}")
        console.print(f"[green]Indexed {result['indexed']} files[/green]")
    if imessage:
        r = requests.post(f"{DAEMON_URL}/index/imessage")
        if r.status_code in (400, 409):
            console.print(f"[red]{r.json().get('detail')}[/red]")
            raise typer.Exit(1)
        r.raise_for_status()
        result = _poll_index_progress("Indexing iMessage")
        console.print(f"[green]Indexed {result['indexed']} conversations[/green]")
    if gcal:
        # OAuth happens on the daemon — if it's the first run, a browser
        # tab will pop while we poll. Print a hint so the user knows what's
        # going on instead of staring at a static progress bar.
        import gcal as gcal_mod
        if not gcal_mod.has_token():
            console.print(
                "[dim]No Google Calendar token cached. A browser window will "
                "open shortly for sign-in.[/dim]"
            )
        r = requests.post(f"{DAEMON_URL}/index/gcal")
        if r.status_code in (400, 409):
            console.print(f"[red]{r.json().get('detail')}[/red]")
            raise typer.Exit(1)
        r.raise_for_status()
        result = _poll_index_progress("Indexing Google Calendar")
        console.print(f"[green]Indexed {result['indexed']} events[/green]")
    if gmail:
        import gmail as gmail_mod
        if not gmail_mod.has_token():
            console.print(
                "[dim]No Gmail token cached. A browser window will "
                "open shortly for sign-in.[/dim]"
            )
        r = requests.post(f"{DAEMON_URL}/index/gmail")
        if r.status_code in (400, 409):
            console.print(f"[red]{r.json().get('detail')}[/red]")
            raise typer.Exit(1)
        r.raise_for_status()
        result = _poll_index_progress("Indexing Gmail")
        console.print(f"[green]Indexed {result['indexed']} messages[/green]")


@app.command()
def reindex(
    path: Path = typer.Argument(..., help="Directory to re-index from scratch."),
):
    """Clear the index and rebuild it from a directory."""
    _ensure_daemon()
    r = requests.post(f"{DAEMON_URL}/clear")
    r.raise_for_status()
    console.print("[yellow]Cleared existing index.[/yellow]")
    resolved = str(path.expanduser().resolve())
    r = requests.post(f"{DAEMON_URL}/index/filesystem", json={"path": resolved})
    r.raise_for_status()
    result = _poll_index_progress(f"Re-indexing {path.name}")
    console.print(f"[green]Re-indexed {result['indexed']} files[/green]")


@app.command("backfill-contacts")
def backfill_contacts():
    """Re-resolve iMessage contact names from the Contacts app for already-
    indexed conversations (no re-embedding needed)."""
    _ensure_daemon()
    r = requests.post(f"{DAEMON_URL}/index/imessage/contacts")
    r.raise_for_status()
    body = r.json()
    console.print(f"[green]Updated display names for {body['chunks_updated']} chunks[/green]")


@app.command()
def status():
    """Show index statistics."""
    _ensure_daemon()
    r = requests.get(f"{DAEMON_URL}/status")
    r.raise_for_status()
    body = r.json()
    n = body.get("total_chunks", 0)
    by_source = body.get("by_source", {})
    console.print(f"[bold]Atlas index[/bold]: {n} chunks stored")
    # Show per-source breakdown including sources with no chunks, so the
    # user can see at a glance what's connected vs missing.
    labels = [
        ("Filesystem", "filesystem"),
        ("iMessage", "imessage"),
        ("Calendar", "gcal"),
        ("Gmail", "gmail"),
    ]
    for label, key in labels:
        count = by_source.get(key, 0)
        if key == "gcal":
            import gcal as gcal_mod
            if count == 0 and not gcal_mod.has_token():
                console.print(f"  {label}: [dim]not connected[/dim]")
                continue
        if key == "gmail":
            import gmail as gmail_mod
            if count == 0 and not gmail_mod.has_token():
                console.print(f"  {label}: [dim]not connected[/dim]")
                continue
        console.print(f"  {label}: {count} chunks")
    if n == 0:
        console.print("[yellow]Nothing indexed yet. Run: python cli.py index <path>[/yellow]")


# ---------------------------------------------------------------------------
# Daemon management
# ---------------------------------------------------------------------------

@daemon_app.command("start")
def daemon_start():
    """Start the daemon (no-op if already running)."""
    if _daemon_alive():
        console.print("[yellow]Daemon already running.[/yellow]")
        return
    log_pos = _log_size()
    _spawn_daemon()
    if _wait_for_daemon(log_start_pos=log_pos):
        console.print(f"[green]Daemon started on {DAEMON_URL}[/green]")
    else:
        console.print(f"[red]Daemon failed to start. Check {DAEMON_LOG_FILE}[/red]")
        raise typer.Exit(1)


def _daemon_pid_by_port() -> Optional[int]:
    """Find the PID of whatever process is bound to DAEMON_PORT (fallback for
    stale PID files)."""
    from config import DAEMON_PORT
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{DAEMON_PORT}", "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return int(out.splitlines()[0]) if out else None
    except (subprocess.CalledProcessError, ValueError):
        return None


@daemon_app.command("stop")
def daemon_stop():
    """Stop the running daemon."""
    pid = _daemon_pid() or _daemon_pid_by_port()
    if pid is None:
        console.print("[yellow]No daemon running.[/yellow]")
        return
    os.kill(pid, signal.SIGTERM)
    DAEMON_PID_FILE.unlink(missing_ok=True)
    console.print(f"[green]Stopped daemon (pid {pid}).[/green]")


@daemon_app.command("restart")
def daemon_restart():
    """Restart the daemon."""
    if _daemon_pid() is not None:
        daemon_stop()
        time.sleep(0.5)
    daemon_start()


@daemon_app.command("logs")
def daemon_logs(
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines to show."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new log lines."),
):
    """Show daemon logs."""
    if not DAEMON_LOG_FILE.exists():
        console.print(f"[yellow]No log file at {DAEMON_LOG_FILE}[/yellow]")
        return
    cmd = ["tail", f"-{tail}"]
    if follow:
        cmd.append("-f")
    cmd.append(str(DAEMON_LOG_FILE))
    subprocess.run(cmd)


# ---------------------------------------------------------------------------
# User config (~/.atlas/config.json) — controls cloud-vs-local query parser.
# Changes require a daemon restart to take effect.
# ---------------------------------------------------------------------------

@config_app.command("show")
def config_show():
    """Print the current user config."""
    cfg = load_user_config()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(table)
    has_key = _has_cloud_api_key()
    console.print(f"[dim]Cloud API key in Keychain: {'yes' if has_key else 'no'}[/dim]")


def _has_cloud_api_key() -> bool:
    try:
        import keyring
        return keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_USERNAME) is not None
    except Exception:
        return False


@config_app.command("set-cloud-parser")
def config_set_cloud_parser(enabled: bool = typer.Argument(..., help="true or false")):
    """Toggle the cloud query parser (true/false). On first enable, prompts for an API key
    and stores it in the macOS Keychain. Restart the daemon to apply."""
    cfg = load_user_config()
    if enabled and not _has_cloud_api_key():
        console.print(
            "[bold]Cloud parser uses an OpenAI-compatible endpoint "
            "(default: Groq, free tier).[/bold]"
        )
        console.print(f"[dim]Endpoint: {cfg['cloud_endpoint']}  Model: {cfg['cloud_model']}[/dim]")
        console.print("Get a free key at https://console.groq.com/keys")
        key = typer.prompt("API key", hide_input=True)
        if not key.strip():
            console.print("[red]Empty key; aborting.[/red]")
            raise typer.Exit(1)
        import keyring
        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_USERNAME, key.strip())
        console.print("[green]Key stored in Keychain.[/green]")
    cfg["cloud_parser"] = enabled
    save_user_config(cfg)
    console.print(f"[green]cloud_parser = {enabled}.[/green] Restart the daemon to apply.")


@config_app.command("clear-cloud-key")
def config_clear_cloud_key():
    """Remove the cloud-parser API key from the Keychain."""
    import keyring
    try:
        keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_USERNAME)
        console.print("[green]Cloud API key removed.[/green]")
    except keyring.errors.PasswordDeleteError:
        console.print("[yellow]No key was stored.[/yellow]")


@config_app.command("set-retrieval-mode")
def config_set_retrieval_mode(
    mode: str = typer.Argument(..., help="One of: chunk, document"),
):
    """Switch the retrieval pipeline between per-chunk embeddings (chunk) and
    per-file LLM-tagged summaries (document). Restart the daemon to apply.

    The two modes use separate SQLite indices (~/.atlas/db/atlas.db and
    ~/.atlas/db/atlas_tagged.db), so switching does not destroy the other
    index. Document mode currently only indexes filesystem content —
    iMessage stays chunk-mode only."""
    if mode not in ("chunk", "document"):
        console.print(f"[red]Invalid mode '{mode}'. Use 'chunk' or 'document'.[/red]")
        raise typer.Exit(1)
    cfg = load_user_config()
    cfg["retrieval_mode"] = mode
    save_user_config(cfg)
    console.print(f"[green]retrieval_mode = {mode}.[/green] Restart the daemon to apply.")
    if mode == "document":
        console.print(
            "[dim]Document mode reads from ~/.atlas/db/atlas_tagged.db. "
            "If this is your first time, run `cli.py reindex <path>` "
            "after the daemon restart to populate it.[/dim]"
        )


@config_app.command("gcal-clear")
def config_gcal_clear():
    """Remove the Google Calendar OAuth token from the Keychain. Useful when
    the 7-day testing-mode refresh window expires, or to force re-auth."""
    import gcal
    if gcal.clear_token():
        console.print("[green]Google Calendar token removed.[/green]")
    else:
        console.print("[yellow]No Google Calendar token was stored.[/yellow]")


@config_app.command("gmail-clear")
def config_gmail_clear():
    """Remove the Gmail OAuth token from the Keychain."""
    import gmail
    if gmail.clear_token():
        console.print("[green]Gmail token removed.[/green]")
    else:
        console.print("[yellow]No Gmail token was stored.[/yellow]")


def _rewrite_leading_help_flag(argv: list[str]) -> list[str]:
    """Allow `cli.py --help <command>` in addition to Typer's native
    `cli.py <command> --help`. Walks past the leading --help/-h and
    collects the following non-flag tokens as the command path
    (e.g. `daemon start`), then re-emits them with --help at the end."""
    if len(argv) < 3 or argv[1] not in ("--help", "-h"):
        return argv
    cmd_path: list[str] = []
    i = 2
    while i < len(argv) and not argv[i].startswith("-"):
        cmd_path.append(argv[i])
        i += 1
    if not cmd_path:
        return argv
    return [argv[0], *cmd_path, "--help", *argv[i:]]


if __name__ == "__main__":
    sys.argv = _rewrite_leading_help_flag(sys.argv)
    app()
