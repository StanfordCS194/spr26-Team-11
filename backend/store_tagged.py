# -*- coding: utf-8 -*-
"""
Document-mode vector store.

Lives in a separate SQLite file (~/.atlas/db/atlas_tagged.db) so it can
coexist with the per-chunk store at ~/.atlas/db/atlas.db. Schema is one
row per file rather than one row per chunk:

    documents(
        source_path TEXT PRIMARY KEY,
        source_type TEXT,         -- "filesystem" | "imessage"
        document_type TEXT,       -- LLM-extracted, e.g. "lecture notes"
        topics TEXT,              -- comma-joined LLM-extracted topics
        path_tokens TEXT,         -- cleaned path tokens (reused from ingest)
        summary TEXT,             -- LLM-extracted 2-3 sentence summary
        summary_embedding BLOB,   -- 384-float vector of the summary
        indexed_at REAL
    )

Retrieval is a single cosine pass over the summary_embedding column (no
re-ranker — there's no per-file batch shape variability and the candidate
set is tiny compared to chunk-mode).
"""
import math
import re
import sqlite3
import time

import numpy as np

from config import TAGGED_DB_FILE


# ---------------------------------------------------------------------------
# Tokenizer used for topic-overlap matching. Splits more aggressively than
# search._tokenize so identifier-style strings get broken into their parts:
#
#   CS107          -> [cs, 107]      (letter-digit boundary)
#   CS_107         -> [cs, 107]      (any non-alphanumeric separator)
#   cs-107         -> [cs, 107]
#   SystemsHomework-> [systems, homework]  (lowercase->uppercase boundary)
#   CSHomework     -> [cshomework]   (runs of capitals stay together)
#
# Both the query and the per-document haystack are tokenized this way so
# the intersection is over comparable token sets.
# ---------------------------------------------------------------------------

_LETTER_DIGIT_RE = re.compile(r'([a-zA-Z])(\d)')
_DIGIT_LETTER_RE = re.compile(r'(\d)([a-zA-Z])')
_LOWER_UPPER_RE = re.compile(r'([a-z])([A-Z])')
_NON_ALNUM_RE = re.compile(r'[^a-zA-Z0-9]+')


def _split_tokens(s: str) -> list[str]:
    """Split `s` into lowercased tokens of length > 1. Splits on any
    non-alphanumeric separator, on letter-digit boundaries, and on
    lowercase->uppercase transitions (camelCase). Runs of consecutive
    uppercase letters stay together — by design CSHomework stays one
    token while SystemsHomework becomes two."""
    s = _LETTER_DIGIT_RE.sub(r'\1 \2', s)
    s = _DIGIT_LETTER_RE.sub(r'\1 \2', s)
    s = _LOWER_UPPER_RE.sub(r'\1 \2', s)
    return [t.lower() for t in _NON_ALNUM_RE.split(s) if len(t) > 1]


# ---------------------------------------------------------------------------
# In-memory cache. Loaded on first query so cosine is a single numpy matmul
# instead of a row-by-row SQL scan. Cleared by add_document() and clear()
# so it gets rebuilt next query — the per-file scale is small (thousands
# of rows, not hundreds of thousands), so rebuild is cheap (~10 ms).
# ---------------------------------------------------------------------------

_cache_paths: list[str] | None = None
_cache_source_types: list[str] | None = None
_cache_embeddings: np.ndarray | None = None
_cache_summaries: list[str] | None = None
_cache_topics: list[str] | None = None  # comma-joined per row
# Per-row set of haystack tokens (topics + source_path + path_tokens),
# precomputed at cache-load using _split_tokens. Used by query() for
# topic-overlap scoring — set intersection rather than substring match.
_cache_haystack_token_sets: list[set[str]] | None = None


def _invalidate_cache() -> None:
    global _cache_paths, _cache_source_types, _cache_embeddings
    global _cache_summaries, _cache_topics, _cache_haystack_token_sets
    _cache_paths = None
    _cache_source_types = None
    _cache_embeddings = None
    _cache_summaries = None
    _cache_topics = None
    _cache_haystack_token_sets = None


def _conn() -> sqlite3.Connection:
    TAGGED_DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(TAGGED_DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            source_path        TEXT PRIMARY KEY,
            source_type        TEXT NOT NULL,
            document_type      TEXT DEFAULT '',
            topics             TEXT DEFAULT '',
            path_tokens        TEXT DEFAULT '',
            summary            TEXT DEFAULT '',
            summary_embedding  BLOB NOT NULL,
            indexed_at         REAL NOT NULL
        )
    """)
    con.commit()
    return con


def _populate_cache(con: sqlite3.Connection) -> None:
    global _cache_paths, _cache_source_types, _cache_embeddings
    global _cache_summaries, _cache_topics, _cache_haystack_token_sets
    rows = con.execute(
        "SELECT source_path, source_type, summary, topics, path_tokens, "
        "summary_embedding FROM documents"
    ).fetchall()
    if not rows:
        _cache_paths = []
        _cache_source_types = []
        _cache_summaries = []
        _cache_topics = []
        _cache_haystack_token_sets = []
        _cache_embeddings = np.zeros((0, 384), dtype=np.float32)
        return
    _cache_paths = [r[0] for r in rows]
    _cache_source_types = [r[1] for r in rows]
    _cache_summaries = [r[2] for r in rows]
    _cache_topics = [r[3] for r in rows]
    # Haystack = topics + path_tokens + source_path. path_tokens has the
    # ingest-time normalized form (e.g. "stanford frosh winter cs 107"),
    # source_path has the raw filesystem path. Both get tokenized the
    # same way as the query so the comparison is apples-to-apples.
    _cache_haystack_token_sets = [
        set(_split_tokens(f"{r[3]} {r[4]} {r[0]}"))
        for r in rows
    ]
    _cache_embeddings = np.stack(
        [np.frombuffer(r[5], dtype=np.float32) for r in rows]
    )


def _ensure_cache() -> None:
    if _cache_paths is not None:
        return
    con = _conn()
    try:
        _populate_cache(con)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def add_document(
    source_path: str,
    source_type: str,
    document_type: str,
    topics: list[str],
    path_tokens: str,
    summary: str,
    summary_embedding: list[float],
) -> None:
    """Upsert one file's tags. Re-indexing a file just overwrites its row."""
    con = _conn()
    try:
        con.execute(
            "INSERT OR REPLACE INTO documents "
            "(source_path, source_type, document_type, topics, path_tokens, "
            " summary, summary_embedding, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_path,
                source_type,
                document_type,
                ", ".join(topics),
                path_tokens,
                summary,
                np.array(summary_embedding, dtype=np.float32).tobytes(),
                time.time(),
            ),
        )
        con.commit()
    finally:
        con.close()
    _invalidate_cache()


def delete_source(source_path: str) -> None:
    con = _conn()
    try:
        con.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
        con.commit()
    finally:
        con.close()
    _invalidate_cache()


def clear() -> None:
    con = _conn()
    try:
        con.execute("DELETE FROM documents")
        con.commit()
    finally:
        con.close()
    _invalidate_cache()


def count() -> int:
    con = _conn()
    try:
        return con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        con.close()


def count_by_source() -> dict[str, int]:
    """Per-source document counts. Used by `cli.py status` to show what's indexed."""
    con = _conn()
    try:
        rows = con.execute(
            "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


# ---------------------------------------------------------------------------
# Read path: cosine over summary embeddings + optional topic-overlap boost.
# Returns "raw" dict in the same shape as store.query()/query_hybrid() so
# search_tagged can format it into Result objects.
# ---------------------------------------------------------------------------

def query(
    embedding: list[float],
    query_tokens: list[str] | None = None,
    n_results: int = 10,
    source_filter: str | None = None,
    topic_weight: float = 0.25,
) -> dict:
    """Cosine search over summary embeddings, optionally re-scored by
    topic-token overlap. Returns the same {documents, metadatas, distances}
    shape as store.query() so callers can reuse formatting code."""
    _ensure_cache()
    assert _cache_embeddings is not None and _cache_paths is not None

    n = _cache_embeddings.shape[0]
    if n == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    q = np.array(embedding, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)
    en = _cache_embeddings / (
        np.linalg.norm(_cache_embeddings, axis=1, keepdims=True) + 1e-12
    )
    cosine_sim = en @ qn                  # [n], higher = more similar
    semantic_distance = 1.0 - cosine_sim   # [n], lower = more similar

    # Topic-overlap boost. Set intersection between the query's tokens and
    # the precomputed haystack token set for each document. sqrt() on the
    # hit ratio gives partial matches meaningful credit — see the comment
    # in search_tagged._adaptive_topic_weight for the design rationale.
    query_set = {t.lower() for t in (query_tokens or []) if t and len(t) > 1}
    if query_set and topic_weight > 0 and _cache_haystack_token_sets is not None:
        topic_score = np.zeros(n, dtype=np.float32)
        for i, haystack_set in enumerate(_cache_haystack_token_sets):
            hits = len(query_set & haystack_set)
            topic_score[i] = math.sqrt(hits / len(query_set))
        distances = (1.0 - topic_weight) * semantic_distance + topic_weight * (1.0 - topic_score)
    else:
        distances = semantic_distance

    if source_filter:
        mask = np.array(
            [st == source_filter for st in (_cache_source_types or [])],
            dtype=bool,
        )
        distances = np.where(mask, distances, np.inf)

    valid_order = np.flatnonzero(np.isfinite(distances))
    if valid_order.size == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    order = valid_order[np.argsort(distances[valid_order])[:n_results]]
    docs = [(_cache_summaries or [""] * n)[i] for i in order]
    metas = [
        {
            "source_type": (_cache_source_types or [""] * n)[i],
            "source_path": (_cache_paths or [""] * n)[i],
            "topics": (_cache_topics or [""] * n)[i],
        }
        for i in order
    ]
    dists = [float(distances[i]) for i in order]
    return {"documents": [docs], "metadatas": [metas], "distances": [dists]}
