import os
import glob
# Pin fastembed's cache to ~/.atlas/models so models survive macOS's
# periodic cleanup of /var/folders/.../T/ (the default tempfile location).
_FASTEMBED_CACHE = os.path.expanduser("~/.atlas/models")
os.environ.setdefault("FASTEMBED_CACHE_PATH", _FASTEMBED_CACHE)
# Only enable offline mode if both fastembed models are already extracted in
# the cache. On a fresh cache (or after a wipe), leave offline mode off so the
# warmup calls in lifespan() can populate it on first run. Detection is
# conservative: we look for ≥2 .onnx files (one per model) anywhere under the
# cache directory.
_cached_onnx = glob.glob(os.path.join(_FASTEMBED_CACHE, "**", "*.onnx"), recursive=True)
if len(_cached_onnx) >= 2:
    # Skip HuggingFace revision check on every model load — saves ~6 min on warm
    # starts. Trade-off: model updates require a manual `huggingface-cli download`.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import embed
import rerank
# Retrieval mode is chosen at daemon startup from ~/.atlas/config.json.
# "chunk" (default): per-chunk embeddings, hybrid retrieval + rerank.
# "document": per-file LLM-generated summary embeddings, separate DB.
# Both subsystems implement the same search/ask/query interface so the
# routes below don't care which is active.
from config import load_user_config
_RETRIEVAL_MODE = load_user_config().get("retrieval_mode", "chunk")
if _RETRIEVAL_MODE == "document":
    import store_tagged as store
    import ingest_tagged as ingest
    import search_tagged as search_mod
else:
    import store
    import ingest
    import search as search_mod


@dataclass
class IndexJob:
    state: str = "idle"          # "idle" | "running" | "done" | "error"
    target: str = ""             # "filesystem:<path>" or "imessage"
    started_at: float = 0.0
    finished_at: float = 0.0
    indexed: int = 0
    total: int = 0               # total files to process (filesystem only); 0 if unknown
    error: str = ""


_index_job = IndexJob()
_index_lock = threading.Lock()

log = logging.getLogger("atlas.daemon")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("retrieval mode: %s", _RETRIEVAL_MODE)
    if len(_cached_onnx) < 2:
        log.info("downloading models on first run (~160 MB, one-time)...")
    log.info("warming fastembed model...")
    embed.embed_one("warmup")
    log.info("warming cross-encoder reranker...")
    rerank.rerank("warmup", ["warmup document"])
    log.info("populating embedding cache (%d chunks)...", store.count())
    con = store._conn()
    store._populate_cache(con)
    con.close()
    # LLM warmup runs in a background thread so /search is reachable
    # immediately. In cloud-parser mode no local model is loaded at all —
    # the first /ask hits the cloud endpoint directly.
    from llm import mode as llm_mode
    if llm_mode() == "local":
        def _warm_llm():
            try:
                from llm import parse_query
                parse_query("warmup")
                log.info("LLM warmup complete (background)")
            except Exception as e:
                log.warning("LLM warmup skipped: %s", e)

        threading.Thread(target=_warm_llm, daemon=True).start()
        log.info("daemon ready (local LLM warming in background)")
    else:
        log.info("daemon ready (cloud parser mode — no local LLM loaded)")
    yield


app = FastAPI(title="Atlas", description="Local AI search across personal data.", lifespan=lifespan)

# Renderer is loaded from http://localhost:5173 in dev (Vite) and from
# file:// in prod-built Electron. Both differ in origin from the daemon at
# 127.0.0.1:8765, so without CORS the browser blocks the cross-origin fetch.
# Allowing localhost-only origins is safe — the daemon already binds to the
# loopback interface.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?|file://.*)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexFilesystemRequest(BaseModel):
    path: str


class SearchResult(BaseModel):
    source_type: str
    source_path: str
    snippet: str
    score: float


class AskRequest(BaseModel):
    query: str
    limit: int = 10


class QueryRequest(BaseModel):
    question: str
    # Default lower than /search/limit=10: too many chunks blow the LLM
    # context window and dilute the answer. 5 is a good RAG default.
    limit: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchResult]


def _run_filesystem_index(path: Path):
    global _index_job

    def on_progress(done: int, total: int):
        with _index_lock:
            _index_job.indexed = done
            _index_job.total = total

    try:
        n = ingest.index_filesystem(path, progress_callback=on_progress)
        with _index_lock:
            _index_job.state = "done"
            _index_job.indexed = n
            _index_job.finished_at = time.time()
    except Exception as e:
        with _index_lock:
            _index_job.state = "error"
            _index_job.error = str(e)
            _index_job.finished_at = time.time()


def _run_imessage_index():
    global _index_job
    try:
        n = ingest.index_imessage()
        with _index_lock:
            _index_job.state = "done"
            _index_job.indexed = n
            _index_job.finished_at = time.time()
    except Exception as e:
        with _index_lock:
            _index_job.state = "error"
            _index_job.error = str(e)
            _index_job.finished_at = time.time()


def _start_job(target: str) -> bool:
    """Atomically start a job. Returns False if one is already running."""
    with _index_lock:
        if _index_job.state == "running":
            return False
        _index_job.state = "running"
        _index_job.target = target
        _index_job.started_at = time.time()
        _index_job.finished_at = 0.0
        _index_job.indexed = 0
        _index_job.error = ""
    return True


@app.post("/index/filesystem")
def index_filesystem(req: IndexFilesystemRequest, background: BackgroundTasks):
    path = Path(req.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    if not _start_job(f"filesystem:{path}"):
        raise HTTPException(status_code=409, detail=f"Index job already running: {_index_job.target}")
    background.add_task(_run_filesystem_index, path)
    return {"status": "started", "target": f"filesystem:{path}"}


@app.post("/index/imessage")
def index_imessage(background: BackgroundTasks):
    if not _start_job("imessage"):
        raise HTTPException(status_code=409, detail=f"Index job already running: {_index_job.target}")
    background.add_task(_run_imessage_index)
    return {"status": "started", "target": "imessage"}


@app.get("/index/status")
def index_status():
    with _index_lock:
        return {
            "state": _index_job.state,
            "target": _index_job.target,
            "indexed": _index_job.indexed,
            "total": _index_job.total,
            "started_at": _index_job.started_at,
            "finished_at": _index_job.finished_at,
            "error": _index_job.error,
        }


@app.get("/search", response_model=list[SearchResult])
def search(
    q: str = Query(..., description="Natural language query"),
    limit: int = 10,
    source: str | None = None,
):
    log.info("searching: %s", q)
    results = search_mod.search(q, n_results=limit, source_filter=source)
    log.info("returned %d results", len(results))
    return [SearchResult(source_type=r.source_type, source_path=r.source_path, snippet=r.snippet, score=r.score) for r in results]


@app.post("/ask", response_model=list[SearchResult])
def ask(req: AskRequest):
    """Conversational query parsed by local LLM."""
    log.info("ask: %s", req.query)
    results = search_mod.ask(req.query, n_results=req.limit)
    log.info("returned %d results", len(results))
    return [SearchResult(source_type=r.source_type, source_path=r.source_path, snippet=r.snippet, score=r.score) for r in results]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """RAG: retrieve top chunks for the question, then have the LLM
    synthesize an answer with [1]/[2]-style citations. Slower than /search
    or /ask because it adds a generation pass after retrieval."""
    log.info("query: %s", req.question)
    out = search_mod.query(req.question, n_results=req.limit)
    sources = [
        SearchResult(
            source_type=r.source_type,
            source_path=r.source_path,
            snippet=r.snippet,
            score=r.score,
        )
        for r in out["sources"]
    ]
    log.info("query: returned %d sources", len(sources))
    return QueryResponse(answer=out["answer"], sources=sources)


@app.post("/clear")
def clear():
    store.clear()
    return {"status": "cleared"}


@app.get("/status")
def status():
    return {"total_chunks": store.count()}


@app.get("/config")
def get_config():
    """Surface the active modes so the UI / clients can react accordingly."""
    from llm import mode as llm_mode
    return {"parser_mode": llm_mode(), "retrieval_mode": _RETRIEVAL_MODE}


if __name__ == "__main__":
    import uvicorn
    from config import DAEMON_HOST, DAEMON_PORT
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=DAEMON_HOST, port=DAEMON_PORT, log_level="info")
