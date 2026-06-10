"""
Lightweight vector store: SQLite for metadata + numpy/scipy for similarity search.
sqlite3 is stdlib; numpy comes with fastembed; scipy is added for sparse matvec
on the path-keyword index.
"""
import sqlite3
import numpy as np
import scipy.sparse as sp
from config import DB_DIR

DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = DB_DIR / "atlas.db"

# ---------------------------------------------------------------------------
# In-memory cache.
# - _cache_embeddings: dense (n_chunks, dim) float32 — for cosine search
# - _cache_meta: list of (id, source_type, source_path, chunk_idx, timestamp,
#                         path_tokens) — lightweight metadata
# - _cache_path_vocab: dict[token -> column index] for the path keyword vocab
# - _cache_path_matrix: sparse (n_chunks, n_vocab) bool — chunk i contains
#                       vocab token j. Replaces the old per-row Python loop in
#                       _path_keyword_score with one sparse matvec at query time.
# Document text stays on disk and is fetched for top-k winners only.
# Populated on first query, invalidated whenever the DB is written.
# ---------------------------------------------------------------------------
_cache_valid = False
_cache_embeddings: np.ndarray | None = None
_cache_meta: list[tuple] | None = None
_cache_path_vocab: dict[str, int] | None = None
_cache_path_matrix: sp.csr_matrix | None = None


def _invalidate_cache() -> None:
    global _cache_valid
    _cache_valid = False


def _build_path_index(ptokens: tuple[str, ...]) -> tuple[dict[str, int], sp.csr_matrix]:
    """Build (vocab, sparse incidence matrix) from per-chunk path-token strings.

    The matrix has one row per chunk, one column per unique path token across
    the index. M[i, j] == 1 iff chunk i's path contains vocab token j.
    """
    vocab: dict[str, int] = {}
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    for chunk_idx, pt in enumerate(ptokens):
        if not pt:
            continue
        seen_in_chunk: set[int] = set()
        for tok in pt.lower().split():
            col = vocab.get(tok)
            if col is None:
                col = len(vocab)
                vocab[tok] = col
            if col not in seen_in_chunk:
                seen_in_chunk.add(col)
                rows_idx.append(chunk_idx)
                cols_idx.append(col)
    n_chunks = len(ptokens)
    n_vocab = len(vocab)
    if n_vocab == 0:
        return vocab, sp.csr_matrix((n_chunks, 0), dtype=np.float32)
    data = np.ones(len(rows_idx), dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (rows_idx, cols_idx)),
        shape=(n_chunks, n_vocab),
        dtype=np.float32,
    )
    return vocab, matrix


def _populate_cache(con: sqlite3.Connection) -> None:
    global _cache_valid, _cache_embeddings, _cache_meta
    global _cache_path_vocab, _cache_path_matrix
    if _cache_valid:
        return
    rows = con.execute(
        "SELECT id, source_type, source_path, chunk_index, "
        "timestamp, embedding, path_tokens, display_name FROM chunks"
    ).fetchall()
    if rows:
        ids, stypes, spaths, cidxs, tss, blobs, ptokens, dnames = zip(*rows)
        _cache_embeddings = np.stack(
            [np.frombuffer(b, dtype=np.float32) for b in blobs]
        )
        _cache_meta = list(zip(ids, stypes, spaths, cidxs, tss, ptokens, dnames))
        _cache_path_vocab, _cache_path_matrix = _build_path_index(ptokens)
    else:
        _cache_embeddings = np.empty((0, 384), dtype=np.float32)
        _cache_meta = []
        _cache_path_vocab = {}
        _cache_path_matrix = sp.csr_matrix((0, 0), dtype=np.float32)
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
            path_tokens  TEXT DEFAULT '',
            display_name TEXT DEFAULT ''
        )
    """)
    con.commit()
    # Migrate existing DBs that predate path_tokens / display_name columns
    for column in ("path_tokens", "display_name"):
        try:
            con.execute(f"ALTER TABLE chunks ADD COLUMN {column} TEXT DEFAULT ''")
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
    display_names: list[str] | None = None,
):
    con = _conn()
    tokens = path_tokens or [""] * len(ids)
    dnames = display_names or [""] * len(ids)
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
            dname,
        )
        for id_, emb, doc, meta, tok, dname in zip(ids, embeddings, documents, metadatas, tokens, dnames)
    ]
    con.executemany(
        "INSERT OR REPLACE INTO chunks "
        "(id, document, source_type, source_path, chunk_index, timestamp, embedding, path_tokens, display_name) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    con.close()
    _invalidate_cache()


def update_display_names(source_type: str, display_names: dict[str, str]) -> int:
    """Backfill display_name for existing rows, keyed by source_path. Cheap
    UPDATE — no re-embedding/re-chunking needed since source_path and chunk
    ids are unaffected. Returns the number of rows updated."""
    con = _conn()
    updated = 0
    for source_path, display_name in display_names.items():
        cur = con.execute(
            "UPDATE chunks SET display_name = ? WHERE source_type = ? AND source_path = ?",
            (display_name, source_type, source_path),
        )
        updated += cur.rowcount
    con.commit()
    con.close()
    if updated:
        _invalidate_cache()
    return updated


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


def distinct_source_paths(source_type: str) -> list[str]:
    con = _conn()
    rows = con.execute(
        "SELECT DISTINCT source_path FROM chunks WHERE source_type = ?",
        (source_type,),
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def count() -> int:
    con = _conn()
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    return n


def count_by_source() -> dict[str, int]:
    """Per-source chunk counts. Used by `cli.py status` to show what's indexed."""
    con = _conn()
    rows = con.execute(
        "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type"
    ).fetchall()
    con.close()
    return dict(rows)


def query(embedding: list[float], n_results: int = 10) -> dict:
    """Pure semantic search — cosine distance only."""
    con = _conn()
    _populate_cache(con)
    con.close()
    return _rank(
        _cache_meta, _cache_embeddings, _cache_path_matrix, _cache_path_vocab,
        embedding, n_results, semantic_weight=1.0, query_tokens=[],
    )


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
        mask_list = [m[1] == source_filter for m in _cache_meta]
        mask = np.array(mask_list, dtype=bool)
        meta = [m for m, keep in zip(_cache_meta, mask_list) if keep]
        emb_matrix = _cache_embeddings[mask]
        path_matrix = _cache_path_matrix[mask] if _cache_path_matrix.shape[0] else _cache_path_matrix
    else:
        meta = _cache_meta
        emb_matrix = _cache_embeddings
        path_matrix = _cache_path_matrix

    return _rank(
        meta, emb_matrix, path_matrix, _cache_path_vocab,
        embedding, n_results, semantic_weight, query_tokens,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _path_keyword_scores_vec(
    query_tokens: list[str],
    path_matrix: sp.csr_matrix,
    vocab: dict[str, int],
) -> np.ndarray:
    """Vectorized path-keyword scoring for ALL chunks at once.

    Returns a numpy array of shape (n_chunks,) where each entry is the fraction
    of query tokens that have at least one substring match against the chunk's
    path tokens. Replaces a Python loop that ran ~1.3M substring comparisons
    per query on a 259K-chunk index.
    """
    n_chunks = path_matrix.shape[0]
    if not query_tokens or not vocab or n_chunks == 0:
        return np.zeros(n_chunks, dtype=np.float32)

    n_q = len(query_tokens)
    n_vocab = len(vocab)

    # Build Q: (n_q, n_vocab) — Q[i, j] = 1 iff query token i substring-matches
    # vocab token j. The substring rule is bidirectional, matching the original
    # _path_keyword_score behavior. The scan over vocab is small (~few thousand)
    # so the Python loop here is acceptable.
    Q = np.zeros((n_q, n_vocab), dtype=np.float32)
    for i, q in enumerate(query_tokens):
        q_lower = q.lower()
        for tok, col in vocab.items():
            if q_lower == tok or q_lower in tok or tok in q_lower:
                Q[i, col] = 1.0

    # path_matrix @ Q.T → (n_chunks, n_q), counts of vocab matches per chunk
    # per query token. Sparse-times-dense matvec, runs in C with SIMD.
    matches = path_matrix @ Q.T
    if sp.issparse(matches):
        matches = matches.toarray()
    has_match = matches > 0  # boolean: did query token i match any path token?
    return has_match.sum(axis=1).astype(np.float32) / n_q


def _rank(
    meta: list[tuple],
    all_emb: np.ndarray,
    path_matrix: sp.csr_matrix,
    path_vocab: dict[str, int],
    embedding: list[float],
    n_results: int,
    semantic_weight: float,
    query_tokens: list[str],
) -> dict:
    if not meta:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    ids, source_types, source_paths, chunk_idxs, timestamps, _path_tokens_list, display_names = zip(*meta)

    q = np.array(embedding, dtype=np.float32)
    norms = np.linalg.norm(all_emb, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-10, norms)
    semantic_dist = 1 - np.dot(all_emb, q) / norms  # lower = better

    if query_tokens and semantic_weight < 1.0:
        path_scores = _path_keyword_scores_vec(query_tokens, path_matrix, path_vocab)
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
            "display_name": display_names[i],
        } for i in top_k]],
        "distances": [[round(float(combined[i]), 4) for i in top_k]],
    }
