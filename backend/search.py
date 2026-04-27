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


def _adaptive_semantic_weight(query_tokens: list[str], source_filter: str | None) -> float:
    """Pick a semantic_weight based on query character.

    - iMessage chunks have empty path_tokens, so path scoring adds no signal → 1.0.
    - Queries with digit-bearing tokens ("cs107", "194w") usually mean a path
      keyword (course code, version) → 0.5 to give path scoring more weight.
    - Plain natural-language queries → 0.7 default.
    """
    if source_filter == "imessage":
        return 1.0
    if any(any(c.isdigit() for c in t) for t in query_tokens):
        return 0.5
    return 0.7


def search(query: str, n_results: int = 10, source_filter: str | None = None) -> list[Result]:
    """Hybrid semantic + path-keyword search.

    Returns at most one result per unique source_path (the best-scoring chunk
    from each file). To get N unique files, the underlying store is asked for
    more chunks than needed and the duplicates are collapsed.
    """
    embedding = embedder.embed_one(query)
    query_tokens = _tokenize(query)
    semantic_weight = _adaptive_semantic_weight(query_tokens, source_filter)
    raw = store.query_hybrid(
        embedding, query_tokens, n_results=n_results * 3,
        source_filter=source_filter, semantic_weight=semantic_weight,
    )

    seen: dict[str, Result] = {}
    for doc, meta, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        path = meta["source_path"]
        if path in seen and seen[path].score <= distance:
            continue
        seen[path] = Result(
            source_type=meta["source_type"],
            source_path=path,
            snippet=doc[:300].replace("\n", " "),
            score=round(distance, 4),
        )

    return sorted(seen.values(), key=lambda x: x.score)[:n_results]


def find_files(query: str, n_results: int = 10, source_filter: str | None = None) -> list[Result]:
    """Alias for search() — search now deduplicates by file."""
    return search(query, n_results=n_results, source_filter=source_filter)


def find_directories(query: str, n_results: int = 10) -> list[Result]:
    """Group search results by parent directory — returns one result per unique directory."""
    raw = search(query, n_results=n_results * 3, source_filter="filesystem")
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
