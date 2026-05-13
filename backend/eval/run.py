"""
Atlas — Search-quality benchmark runner.

Calls each query in queries.py against the running daemon, records the
rank of the expected match (0-indexed, -1 if missing from top-10), and
records HTTP roundtrip latency. Prints a summary table; optionally saves
the full result to a timestamped JSON file under eval/runs/ so future
changes can be A/B-compared against a stored baseline.

Usage (from the backend directory, with the daemon running):
  .venv/bin/python eval/run.py
  .venv/bin/python eval/run.py --label baseline
  .venv/bin/python eval/run.py --label with-hyde
  .venv/bin/python eval/run.py --no-save        # don't write to runs/

Comparing two runs:
  diff eval/runs/2026-05-06-baseline.json eval/runs/2026-05-06-with-hyde.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

# eval/run.py imports a sibling module — make the eval/ dir importable
# regardless of where the user invokes the script from.
sys.path.insert(0, str(Path(__file__).parent))
from queries import QUERIES  # noqa: E402

DAEMON_URL = "http://127.0.0.1:8765"


# ---------------------------------------------------------------------------
# Daemon calls
# ---------------------------------------------------------------------------

def _call_search(params: dict) -> tuple[list, float]:
    t0 = time.perf_counter()
    r = requests.get(f"{DAEMON_URL}/search", params=params, timeout=30)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


def _call_ask(body: dict) -> tuple[list, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{DAEMON_URL}/ask", json=body, timeout=60)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), elapsed


def _check_daemon() -> None:
    try:
        r = requests.get(f"{DAEMON_URL}/status", timeout=2)
        r.raise_for_status()
    except requests.RequestException as e:
        print(
            f"[eval] Daemon at {DAEMON_URL} is not reachable: {e}\n"
            "       Start it with: .venv/bin/python cli.py daemon start",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Match predicates
# ---------------------------------------------------------------------------

def _expand(p: str) -> str:
    """User-friendly path normalisation: expand ~ and trim trailing /."""
    return str(Path(p).expanduser()).rstrip("/")


def _find_rank(results: list[dict], match: dict) -> int:
    """Return 0-indexed rank of the first result matching the predicate, or -1.

    Match types are documented in queries.py.
    """
    mtype = match["type"]
    value = match["value"]

    if mtype == "path_exact":
        target = _expand(value)
        for i, r in enumerate(results):
            if _expand(r["source_path"]) == target:
                return i
    elif mtype == "path_startswith":
        target = _expand(value)
        for i, r in enumerate(results):
            if _expand(r["source_path"]).startswith(target):
                return i
    elif mtype == "snippet_contains":
        needle = value.lower()
        for i, r in enumerate(results):
            if needle in r.get("snippet", "").lower():
                return i
    else:
        raise ValueError(f"Unknown match type: {mtype}")

    return -1


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def run() -> list[dict]:
    """Execute every query and collect a uniform summary record per query."""
    summary = []
    for q in QUERIES:
        record = {
            "id": q["id"],
            "endpoint": q["endpoint"],
            "rank": -1,
            "latency_ms": 0.0,
            "n_results": 0,
            "top_paths": [],
            "error": None,
        }
        try:
            if q["endpoint"] == "search":
                results, elapsed = _call_search(q["params"])
            elif q["endpoint"] == "ask":
                results, elapsed = _call_ask(q["params"])
            else:
                raise ValueError(f"Unknown endpoint: {q['endpoint']}")
            record["latency_ms"] = round(elapsed * 1000, 1)
            record["n_results"] = len(results)
            record["top_paths"] = [r["source_path"] for r in results[:3]]
            record["rank"] = _find_rank(results, q["match"])
        except Exception as e:
            record["error"] = str(e)
        summary.append(record)
    return summary


def print_table(summary: list[dict]) -> None:
    print(f"\n{'ID':<22} {'Endpoint':<8} {'Rank':>6} {'Latency':>10}  {'Note'}")
    print("-" * 90)
    for s in summary:
        rank_str = str(s["rank"]) if s["rank"] >= 0 else "MISS"
        latency = f"{s['latency_ms']}ms"
        note = s["error"] if s["error"] else f"top: {Path(s['top_paths'][0]).name if s['top_paths'] else '(empty)'}"
        print(f"{s['id']:<22} {s['endpoint']:<8} {rank_str:>6} {latency:>10}  {note}")

    found = [s for s in summary if s["rank"] >= 0]
    n = len(summary)
    print()
    print(f"Found {len(found)}/{n} expected matches in top-10")
    if found:
        avg_rank = sum(s["rank"] for s in found) / len(found)
        print(f"Average rank when found: {avg_rank:.1f}  (lower = better, 0 = top hit)")
    if any(s["latency_ms"] for s in summary):
        latencies = [s["latency_ms"] for s in summary if s["latency_ms"]]
        print(f"Median latency: {sorted(latencies)[len(latencies)//2]:.1f} ms")


def save(summary: list[dict], label: str | None) -> Path:
    """Persist the run as JSON for later A/B comparison."""
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H%M%S")
    fname = f"{stamp}-{label}.json" if label else f"{stamp}.json"
    path = runs_dir / fname
    path.write_text(json.dumps(summary, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument(
        "--label",
        help="Optional label appended to the saved-run filename (e.g. 'baseline').",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't persist the run to eval/runs/ — useful for ad-hoc checks.",
    )
    args = parser.parse_args()

    _check_daemon()
    summary = run()
    print_table(summary)
    if not args.no_save:
        path = save(summary, args.label)
        print(f"\nSaved to: {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
