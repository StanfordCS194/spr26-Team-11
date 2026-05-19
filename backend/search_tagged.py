# -*- coding: utf-8 -*-
"""
Document-mode retrieval. Public interface mirrors search.py so main.py
can pick a module via `import search_tagged as search_mod`.

No cross-encoder rerank — there's only one row per file and the candidate
set is two orders of magnitude smaller than chunk-mode, so the bi-encoder
+ topic-overlap score is enough. Saves the ~700–1200 ms rerank cost from
chunk-mode search.
"""
import re
from pathlib import Path

import embed as embedder
import store_tagged

# Reuse the dataclass + tokenization from chunk-mode so the HTTP layer
# doesn't need to know which retrieval module is active.
from search import Result, _tokenize


def search(
    query: str,
    n_results: int = 10,
    source_filter: str | None = None,
) -> list[Result]:
    """One-stage retrieval: cosine over summary embeddings, optionally
    boosted by topic-token overlap. Returns Result objects compatible
    with the existing /search response shape."""
    embedding = embedder.embed_one(query)
    query_tokens = _tokenize(query)

    raw = store_tagged.query(
        embedding=embedding,
        query_tokens=query_tokens,
        n_results=n_results,
        source_filter=source_filter,
    )

    results: list[Result] = []
    for doc, meta, distance in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        # Snippet is the summary (max ~300 chars to keep parity with
        # chunk-mode's snippet shape); `text` holds the full summary so
        # /query can feed it to the LLM as RAG context.
        snippet = (doc or "").replace("\n", " ")
        results.append(Result(
            source_type=meta["source_type"],
            source_path=meta["source_path"],
            snippet=snippet[:300],
            score=round(distance, 4),
            text=doc or "",
        ))
    return results


def find_files(
    query: str,
    n_results: int = 10,
    source_filter: str | None = None,
) -> list[Result]:
    """Same as search() in document mode — every row is already a file."""
    return search(query, n_results=n_results, source_filter=source_filter)


def find_directories(query: str, n_results: int = 10) -> list[Result]:
    """Group results by parent directory. Each unique directory gets the
    best (lowest-distance) score from the files inside it."""
    raw = search(query, n_results=n_results * 3, source_filter="filesystem")
    by_dir: dict[str, float] = {}
    for r in raw:
        directory = str(Path(r.source_path).parent)
        if directory not in by_dir or r.score < by_dir[directory]:
            by_dir[directory] = r.score

    return [
        Result(
            source_type="filesystem",
            source_path=directory,
            snippet=f"Directory: {directory}",
            score=round(score, 4),
        )
        for directory, score in sorted(by_dir.items(), key=lambda x: x[1])[:n_results]
    ]


def ask(user_input: str, n_results: int = 10) -> list[Result]:
    """Conversational entry point — same intent routing as chunk-mode."""
    import logging
    from llm import parse_query
    log = logging.getLogger("atlas.daemon")
    log.info("parsing query with LLM...")
    try:
        parsed = parse_query(user_input)
    except Exception as e:
        log.warning("LLM parse failed (%s); falling back to plain search", e)
        return search(user_input, n_results=n_results)
    log.info("parsed intent=%s terms=%r source=%s",
             parsed.intent, parsed.search_terms, parsed.source_filter)

    if parsed.intent == "find_directory":
        return find_directories(parsed.search_terms, n_results=n_results)
    if parsed.intent == "find_file":
        return find_files(parsed.search_terms, n_results=n_results,
                          source_filter=parsed.source_filter)
    return search(parsed.search_terms, n_results=n_results,
                  source_filter=parsed.source_filter)


# RAG /query: same structure as search.query but draws its context from the
# tagged store. Each "context" is a file's full summary — much shorter than
# a raw chunk, so we can fit more sources per call.
_QUERY_CHUNK_CHARS = 1000


def query(question: str, n_results: int = 5) -> dict:
    """RAG: retrieve top files by summary, then synthesize an answer."""
    import logging
    from llm import synthesize_answer
    log = logging.getLogger("atlas.daemon")
    log.info("query (tagged): retrieving for %r", question)
    sources = search(question, n_results=n_results)

    if not sources:
        return {"answer": "No relevant content found in the index.", "sources": []}

    contexts = [
        {"source_path": r.source_path, "text": r.text[:_QUERY_CHUNK_CHARS]}
        for r in sources
    ]
    log.info("query (tagged): synthesizing from %d sources", len(sources))
    try:
        answer = synthesize_answer(question, contexts).strip()
    except Exception as e:
        log.warning("query: LLM synthesis failed (%s); returning sources only", e)
        answer = (
            "Couldn't generate an answer (LLM unavailable). "
            "Top relevant sources are listed below."
        )
    return {"answer": answer, "sources": sources}
