"""
Lightweight vector store: SQLite for metadata + numpy for similarity search.
No external vector DB dependency — sqlite3 is stdlib, numpy comes with sentence-transformers.
"""
import sqlite3
import numpy as np
from config import DB_DIR

DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = DB_DIR / "atlas.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_DB_PATH)
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
    return con


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


def _load_all(con: sqlite3.Connection):
    return con.execute(
        "SELECT id, document, source_type, source_path, chunk_index, timestamp, embedding, path_tokens "
        "FROM chunks"
    ).fetchall()


def query(embedding: list[float], n_results: int = 10) -> dict:
    """Pure semantic search — cosine distance only."""
    con = _conn()
    rows = _load_all(con)
    con.close()
    return _rank(rows, embedding, n_results, semantic_weight=1.0, query_tokens=[])


def query_hybrid(
    embedding: list[float],
    query_tokens: list[str],
    n_results: int = 10,
    semantic_weight: float = 0.7,
) -> dict:
    """Hybrid search: weighted combination of semantic similarity and path keyword overlap."""
    con = _conn()
    rows = _load_all(con)
    con.close()
    return _rank(rows, embedding, n_results, semantic_weight, query_tokens)


def _rank(rows, embedding, n_results, semantic_weight, query_tokens) -> dict:
    if not rows:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    ids, docs, source_types, source_paths, chunk_idxs, timestamps, blobs, path_tokens_list = zip(*rows)

    # Semantic score (cosine distance)
    all_emb = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    q = np.array(embedding, dtype=np.float32)
    norms = np.linalg.norm(all_emb, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-10, norms)
    semantic_dist = 1 - np.dot(all_emb, q) / norms  # lower = better

    # Path keyword score
    if query_tokens and semantic_weight < 1.0:
        query_set = set(t.lower() for t in query_tokens)
        path_scores = np.array([
            _path_keyword_score(query_set, pt) for pt in path_tokens_list
        ])
        path_dist = 1 - path_scores  # lower = better
        combined = semantic_weight * semantic_dist + (1 - semantic_weight) * path_dist
    else:
        combined = semantic_dist

    k = min(n_results, len(rows))
    top_k = np.argsort(combined)[:k]

    return {
        "ids": [[ids[i] for i in top_k]],
        "documents": [[docs[i] for i in top_k]],
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


def count() -> int:
    con = _conn()
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    con.close()
    return n


def delete_source(source_type: str, source_path: str):
    con = _conn()
    con.execute(
        "DELETE FROM chunks WHERE source_type = ? AND source_path = ?",
        (source_type, source_path),
    )
    con.commit()
    con.close()


def clear():
    con = _conn()
    con.execute("DELETE FROM chunks")
    con.commit()
    con.close()
