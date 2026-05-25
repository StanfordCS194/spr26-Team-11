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

# Reuse the dataclass from chunk-mode so the HTTP layer doesn't need to
# know which retrieval module is active. Tokenization here uses the
# richer `_split_tokens` (splits CS107 -> cs, 107; SystemsHomework ->
# systems, homework) instead of chunk-mode's plain `_tokenize`, because
# document-mode topic-overlap is set intersection — it benefits from the
# same identifier-aware splits on both sides.
from search import Result
from store_tagged import _split_tokens


# Normalize identifier-shaped fragments in the query before it goes into
# the bi-encoder so "cs107", "cs_107", and "cs-107" all embed identically
# to "cs 107". This is a *minimal* normalization compared to _split_tokens
# — case and length-1 tokens are preserved so the embedder still sees the
# original natural-language structure for everything else.
_EMBED_LD_RE = re.compile(r'([a-zA-Z])(\d)')
_EMBED_DL_RE = re.compile(r'(\d)([a-zA-Z])')
_EMBED_SEP_RE = re.compile(r'[_\-]+')


def _normalize_for_embedding(query: str) -> str:
    """Insert spaces at letter-digit boundaries and replace _ / - with
    spaces. Keeps case, doesn't drop tokens — bi-encoder still sees the
    natural-language form, just with identifier-style runs broken up the
    same way the indexed topics + summary text describes them."""
    q = _EMBED_LD_RE.sub(r'\1 \2', query)
    q = _EMBED_DL_RE.sub(r'\1 \2', q)
    q = _EMBED_SEP_RE.sub(' ', q)
    return q


def _adaptive_topic_weight(query_tokens: list[str], source_filter: str | None) -> float:
    """Pick a topic_weight in [0.0, 0.5] from query character.

    Mirrors search._adaptive_semantic_weight in spirit but inverted:
    here `topic_weight` is the positive coefficient on the topic-overlap
    side of the blend, so we *raise* it for identifier-shaped queries.

    - Sources where path/topic carries no signal (imessage, gcal)
      → 0.0 (semantic only).
    - Digit-bearing token suggests a course code / version / project ID
      → +0.15 (catches "cs107", "194w", "v2.3" generically without
      requiring a hardcoded vocabulary).
    - Short query (<= 2 split tokens) suggests the user is being specific
      → +0.10. Counted in _split_tokens output so "cs107 homework" and
      "cs 107 homework" yield identical weights (both produce 3 tokens
      and both miss the bonus). The threshold catches "machine learning"
      (2 tokens) without including "cs107 homework" (3).
    - Base → 0.25, ceiling 0.5.
    """
    if source_filter in ("imessage", "gcal"):
        return 0.0
    w = 0.25
    if any(any(c.isdigit() for c in t) for t in query_tokens):
        w += 0.15
    if len(query_tokens) <= 2:
        w += 0.10
    return min(0.5, w)


def search(
    query: str,
    n_results: int = 10,
    source_filter: str | None = None,
) -> list[Result]:
    """One-stage retrieval: cosine over summary embeddings, boosted by
    topic-token overlap (set intersection over identifier-aware tokens,
    sqrt-shaped credit, query-aware topic_weight)."""
    embedding = embedder.embed_one(_normalize_for_embedding(query))
    query_tokens = _split_tokens(query)
    topic_weight = _adaptive_topic_weight(query_tokens, source_filter)

    raw = store_tagged.query(
        embedding=embedding,
        query_tokens=query_tokens,
        n_results=n_results,
        source_filter=source_filter,
        topic_weight=topic_weight,
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
