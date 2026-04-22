import re
from dataclasses import dataclass
from pathlib import Path

import embed as embedder
import store


@dataclass
class Result:
    source_type: str
    source_path: str
    snippet: str
    score: float  # lower = more similar (combined distance)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if len(t) > 1]


def search(query: str, n_results: int = 10, source_filter: str | None = None) -> list[Result]:
    """Hybrid semantic + path-keyword search."""
    if store.count() == 0:
        return []

    embedding = embedder.embed_one(query)
    query_tokens = _tokenize(query)
    raw = store.query_hybrid(embedding, query_tokens, n_results=n_results * 2)

    results = []
    for doc, meta, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        if source_filter and meta["source_type"] != source_filter:
            continue
        results.append(Result(
            source_type=meta["source_type"],
            source_path=meta["source_path"],
            snippet=doc[:300].replace("\n", " "),
            score=round(distance, 4),
        ))

    return results[:n_results]


def find_files(query: str, n_results: int = 10, source_filter: str | None = None) -> list[Result]:
    """Search and deduplicate by file — returns one result per unique source path."""
    raw = search(query, n_results=n_results * 3, source_filter=source_filter)
    seen: dict[str, Result] = {}
    for r in raw:
        if r.source_path not in seen or r.score < seen[r.source_path].score:
            seen[r.source_path] = r
    return sorted(seen.values(), key=lambda x: x.score)[:n_results]


def find_directories(query: str, n_results: int = 10) -> list[Result]:
    """Search and group by parent directory — returns one result per unique directory."""
    raw = search(query, n_results=n_results * 5, source_filter="filesystem")
    seen: dict[str, float] = {}
    for r in raw:
        directory = str(Path(r.source_path).parent)
        if directory not in seen or r.score < seen[directory]:
            seen[directory] = r.score

    return [
        Result(
            source_type="filesystem",
            source_path=directory,
            snippet=f"Directory: {directory}",
            score=round(score, 4),
        )
        for directory, score in sorted(seen.items(), key=lambda x: x[1])[:n_results]
    ]


def ask(user_input: str, n_results: int = 10) -> list[Result]:
    """Conversational query: parse intent with local LLM, then route to the right search mode."""
    from llm import parse_query
    parsed = parse_query(user_input)

    if parsed.intent == "find_directory":
        return find_directories(parsed.search_terms, n_results=n_results)

    if parsed.intent == "find_file":
        return find_files(parsed.search_terms, n_results=n_results, source_filter=parsed.source_filter)

    # Default: content search
    return search(parsed.search_terms, n_results=n_results, source_filter=parsed.source_filter)
