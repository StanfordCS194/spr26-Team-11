import os
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
from pydantic import BaseModel

import store
import ingest
import embed
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
    log.info("warming fastembed model...")
    embed.embed_one("warmup")
    log.info("populating embedding cache (%d chunks)...", store.count())
    con = store._conn()
    store._populate_cache(con)
    con.close()
    log.info("warming LLM (Phi-3-mini)...")
    try:
        from llm import parse_query
        parse_query("warmup")
    except Exception as e:
        log.warning("LLM warmup skipped: %s", e)
    log.info("daemon ready")
    yield


app = FastAPI(title="Atlas", description="Local AI search across personal data.", lifespan=lifespan)


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
    results = search_mod.search(q, n_results=limit, source_filter=source)
    return [SearchResult(source_type=r.source_type, source_path=r.source_path, snippet=r.snippet, score=r.score) for r in results]


@app.post("/ask", response_model=list[SearchResult])
def ask(req: AskRequest):
    """Conversational query parsed by local LLM."""
    results = search_mod.ask(req.query, n_results=req.limit)
    return [SearchResult(source_type=r.source_type, source_path=r.source_path, snippet=r.snippet, score=r.score) for r in results]


@app.post("/clear")
def clear():
    store.clear()
    return {"status": "cleared"}


@app.get("/status")
def status():
    return {"total_chunks": store.count()}


if __name__ == "__main__":
    import uvicorn
    from config import DAEMON_HOST, DAEMON_PORT
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=DAEMON_HOST, port=DAEMON_PORT, log_level="info")
