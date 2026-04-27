"""
Lightweight vector store: SQLite for metadata + numpy for similarity search.
No external vector DB dependency — sqlite3 is stdlib, numpy comes with fastembed.
"""
import sqlite3
import numpy as np
from config import DB_DIR

DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = DB_DIR / "atlas.db"

# ---------------------------------------------------------------------------
# In-memory embedding cache (split: lightweight metadata + embeddings only).
# Document text stays on disk and is fetched for top-k winners only — keeps
# the cache footprint dominated by embeddings (~1.5 KB/chunk) instead of
# embeddings + raw text (~5 KB/chunk).
# Populated on first query, invalidated whenever the DB is written.
# _cache_meta rows: (id, source_type, source_path, chunk_idx, timestamp, path_tokens)
# ---------------------------------------------------------------------------
_cache_valid = False
_cache_embeddings: np.ndarray | None = None  # shape (N, dim)
_cache_meta: list[tuple] | None = None


def _invalidate_cache() -> None:
    global _cache_valid
    _cache_valid = False


def _populate_cache(con: sqlite3.Connection) -> None:
    global _cache_valid, _cache_embeddings, _cache_meta
    if _cache_valid:
        return
    rows = con.execute(
        "SELECT id, source_type, source_path, chunk_index, "
        "timestamp, embedding, path_tokens FROM chunks"
    ).fetchall()
    if rows:
        ids, stypes, spaths, cidxs, tss, blobs, ptokens = zip(*rows)
        _cache_embeddings = np.stack(
            [np.frombuffer(b, dtype=np.float32) for b in blobs]
        )
        _cache_meta = list(zip(ids, stypes, spaths, cidxs, tss, ptokens))
    else:
        _cache_embeddings = np.empty((0, 384), dtype=np.float32)
        _cache_meta = []
    _cache_valid = True


def _fetch_documents(ids: list[str]) -> dict[str, str]:
    """Fetch chunk text for the given ids — one round-trip, primary-key lookup."""
    if not ids:
        return {}
    con = _conn()
    placeholders = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT id, document FROM chunks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    con.close()
    return dict(rows)


# ---------------------------------------------------------------------------
# Schema init (called once per connection)
# ---------------------------------------------------------------------------

_schema_initialized = False


def _conn() -> sqlite3.Connection:
    global _schema_initialized
    con = sqlite3.connect(_DB_PATH)
    if _schema_initialized:
        return con
    con.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           TEXT PRIMARY KEY,
            document     TEXT NOT NULL,
            source_type  TEXT NOT NULL,
            source_path  TEXT NOT NULL,
            chunk_index  INTEGER NOT NULL,
            timestamp    TEXT,
            embedding    BLOB NOT NULL,
            path_tokens  TEXT DEFAULT ''
        )
    """)
    con.commit()
    # Migrate existing DBs that predate path_tokens column
    try:
        con.execute("ALTER TABLE chunks ADD COLUMN path_tokens TEXT DEFAULT ''")
        con.commit()
    except sqlite3.OperationalError:
        pass  # column already exists
    _schema_initialized = True
    return con


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def add(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
    path_tokens: list[str] | None = None,
):
    con = _conn()
    tokens = path_tokens or [""] * len(ids)
    rows = [
        (
            id_,
            doc,
            meta["source_type"],
            meta["source_path"],
            meta["chunk_index"],
            meta.get("timestamp", ""),
            np.array(emb, dtype=np.float32).tobytes(),
            tok,
        )
        for id_, emb, doc, meta, tok in zip(ids, embeddings, documents, metadatas, tokens)
    ]
    con.executemany(
        "INSERT OR REPLACE INTO chunks "
        "(id, document, source_type, source_path, chunk_index, timestamp, embedding, path_tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    _invalidate_cache()


def delete_source(source_type: str, source_path: str):
    con = _conn()
    con.execute(
        "DELETE FROM chunks WHERE source_type = ? AND source_path = ?",
        (source_type, source_path),
    )
    con.commit()
    con.close()
    _invalidate_cache()


def clear():
    con = _conn()
    con.execute("DELETE FROM chunks")
    con.commit()
    con.close()
    _invalidate_cache()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def count() -> int:
    con = _conn()
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    return n


def query(embedding: list[float], n_results: int = 10) -> dict:
    """Pure semantic search — cosine distance only."""
    con = _conn()
    _populate_cache(con)
    con.close()
    return _rank(_cache_meta, _cache_embeddings, embedding, n_results,
                 semantic_weight=1.0, query_tokens=[])


def query_hybrid(
    embedding: list[float],
    query_tokens: list[str],
    n_results: int = 10,
    semantic_weight: float = 0.7,
    source_filter: str | None = None,
) -> dict:
    """Hybrid search: weighted combination of semantic similarity and path keyword overlap."""
    con = _conn()
    _populate_cache(con)
    con.close()

    if source_filter and _cache_meta:
        mask = np.array([m[1] == source_filter for m in _cache_meta], dtype=bool)
        meta = [m for m, keep in zip(_cache_meta, mask) if keep]
        emb_matrix = _cache_embeddings[mask]
    else:
        meta = _cache_meta
        emb_matrix = _cache_embeddings

    return _rank(meta, emb_matrix, embedding, n_results, semantic_weight, query_tokens)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _rank(
    meta: list[tuple],
    all_emb: np.ndarray,
    embedding: list[float],
    n_results: int,
    semantic_weight: float,
    query_tokens: list[str],
) -> dict:
    if not meta:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    ids, source_types, source_paths, chunk_idxs, timestamps, path_tokens_list = zip(*meta)

    q = np.array(embedding, dtype=np.float32)
    norms = np.linalg.norm(all_emb, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-10, norms)
    semantic_dist = 1 - np.dot(all_emb, q) / norms  # lower = better

    if query_tokens and semantic_weight < 1.0:
        query_set = set(t.lower() for t in query_tokens)
        path_scores = np.array([
            _path_keyword_score(query_set, pt) for pt in path_tokens_list
        ])
        combined = semantic_weight * semantic_dist + (1 - semantic_weight) * (1 - path_scores)
    else:
        combined = semantic_dist

    k = min(n_results, len(meta))
    top_k = np.argsort(combined)[:k]

    top_ids = [ids[i] for i in top_k]
    docs = _fetch_documents(top_ids)

    return {
        "ids": [top_ids],
        "documents": [[docs.get(id_, "") for id_ in top_ids]],
        "metadatas": [[{
            "source_type": source_types[i],
            "source_path": source_paths[i],
            "chunk_index": chunk_idxs[i],
            "timestamp": timestamps[i],
        } for i in top_k]],
        "distances": [[round(float(combined[i]), 4) for i in top_k]],
    }


def _path_keyword_score(query_set: set[str], path_tokens_str: str) -> float:
    """Fraction of query tokens that match any path token.

    Uses substring matching in both directions so that e.g. query token "cs107"
    matches path tokens "cs" and "107" (from a path like "CS 107/"), and query
    token "cs" matches a path token "cs194w".
    """
    if not query_set or not path_tokens_str:
        return 0.0
    path_tokens = path_tokens_str.lower().split()

    def _matches(q: str, p: str) -> bool:
        return q == p or q in p or p in q

    matched = sum(
        1 for q in query_set if any(_matches(q, p) for p in path_tokens)
    )
    return matched / len(query_set)
