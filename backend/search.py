import math
import re
from dataclasses import dataclass
from pathlib import Path

import embed as embedder
import rerank as reranker
import store

# How many candidates to fetch from the bi-encoder before re-ranking.
# Larger = better recall going into the cross-encoder, slower per query.
RERANK_CANDIDATES = 50


@dataclass
class Result:
    source_type: str
    source_path: str
    snippet: str
    score: float  # lower = more similar (after rerank: 1 - sigmoid(rerank_logit))
    # Full chunk text. Used by the /query RAG path which needs more than
    # the 300-char snippet to feed the LLM. Not serialized over /search.
    text: str = ""


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r'[^a-zA-Z0-9]+', text.lower()) if len(t) > 1]


def _adaptive_semantic_weight(query_tokens: list[str], source_filter: str | None) -> float:
    """Pick a semantic_weight based on query character.

    - iMessage and Calendar chunks have empty path_tokens, so path scoring
      adds no signal → 1.0.
    - Queries with digit-bearing tokens ("cs107", "194w") usually mean a path
      keyword (course code, version) → 0.5 to give path scoring more weight.
    - Plain natural-language queries → 0.7 default.
    """
    if source_filter in ("imessage", "gcal"):
        return 1.0
    if any(any(c.isdigit() for c in t) for t in query_tokens):
        return 0.5
    return 0.7


def search(query: str, n_results: int = 10, source_filter: str | None = None) -> list[Result]:
    """Two-stage search: bi-encoder retrieval → cross-encoder re-ranking.

    Stage 1 (cheap, recall-oriented): hybrid semantic + path-keyword retrieval
    fetches RERANK_CANDIDATES chunks, deduplicated to at most one per file.

    Stage 2 (expensive, precision-oriented): the cross-encoder scores each
    deduped (query, document) pair jointly and provides the final ordering.

    Returns at most n_results results.
    """
    embedding = embedder.embed_one(query)
    query_tokens = _tokenize(query)
    semantic_weight = _adaptive_semantic_weight(query_tokens, source_filter)

    # Over-fetch to give the reranker a wide candidate pool. The bi-encoder is
    # cheap enough that fetching 50 instead of 30 is negligible.
    fetch_n = max(n_results * 3, RERANK_CANDIDATES)
    raw = store.query_hybrid(
        embedding, query_tokens, n_results=fetch_n,
        source_filter=source_filter, semantic_weight=semantic_weight,
    )

    # Dedup by source_path, keeping the lowest-distance chunk per file.
    # Carry the full doc text through for re-ranking.
    seen: dict[str, tuple[str, dict, float]] = {}
    for doc, meta, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        path = meta["source_path"]
        if path in seen and seen[path][2] <= distance:
            continue
        seen[path] = (doc, meta, distance)

    if not seen:
        return []

    items = list(seen.values())
    docs = [doc for doc, _, _ in items]
    rerank_scores = reranker.rerank(query, docs)

    # Convert reranker logits to a [0, 1] distance (lower = better) so the
    # field stays consistent with the rest of the system.
    results = [
        Result(
            source_type=meta["source_type"],
            source_path=meta["source_path"],
            snippet=doc[:300].replace("\n", " "),
            score=round(1.0 / (1.0 + math.exp(rs)), 4),
            text=doc,
        )
        for (doc, meta, _), rs in zip(items, rerank_scores)
    ]

    results.sort(key=lambda r: r.score)
    return results[:n_results]


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
    """Conversational query: parse intent with the LLM, then route to the
    right search mode. On any LLM failure (network down for cloud mode,
    model load error for local, malformed output, timeout) fall back to a
    plain content search with the raw input — same as the parser's own
    schema-validation fallback path."""
    import logging
    from llm import parse_query
    log = logging.getLogger("atlas.daemon")
    log.info("parsing query with LLM...")
    try:
        parsed = parse_query(user_input)
    except Exception as e:
        log.warning("LLM parse failed (%s); falling back to plain search", e)
        return search(user_input, n_results=n_results)
    log.info("parsed intent=%s terms=%r source=%s", parsed.intent, parsed.search_terms, parsed.source_filter)

    if parsed.intent == "find_directory":
        return find_directories(parsed.search_terms, n_results=n_results)

    if parsed.intent == "find_file":
        return find_files(parsed.search_terms, n_results=n_results, source_filter=parsed.source_filter)

    # Default: content search
    return search(parsed.search_terms, n_results=n_results, source_filter=parsed.source_filter)


# Cap each retrieved chunk's contribution to the RAG prompt. Matches the
# guideline in llm._RAG_CHUNK_CHARS — defined here so search.py doesn't
# need to import private llm constants.
_QUERY_CHUNK_CHARS = 800


def query(question: str, n_results: int = 5) -> dict:
    """RAG: retrieve top chunks for `question`, then have the LLM synthesize
    an answer from them. Returns:

        {"answer": str, "sources": list[Result]}

    The sources are the same shape as /search returns. If the LLM call fails
    for any reason, returns the sources alone with a fallback `answer`
    explaining that synthesis was unavailable — the user still gets ranked
    results, just without the synthesized prose."""
    import logging
    from llm import synthesize_answer
    log = logging.getLogger("atlas.daemon")
    log.info("query: retrieving for %r", question)
    sources = search(question, n_results=n_results)

    if not sources:
        return {"answer": "No relevant content found in the index.", "sources": []}

    contexts = [
        {"source_path": r.source_path, "text": r.text[:_QUERY_CHUNK_CHARS]}
        for r in sources
    ]
    log.info("query: synthesizing answer from %d sources", len(sources))
    try:
        answer = synthesize_answer(question, contexts).strip()
    except Exception as e:
        log.warning("query: LLM synthesis failed (%s); returning sources only", e)
        answer = (
            "Couldn't generate an answer (LLM unavailable). "
            "Top relevant sources are listed below."
        )
    return {"answer": answer, "sources": sources}
