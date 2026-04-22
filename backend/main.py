from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel

import store
import ingest
import search as search_mod

app = FastAPI(title="Atlas", description="Local AI search across personal data.")


class IndexFilesystemRequest(BaseModel):
    path: str


class IndexResponse(BaseModel):
    indexed: int


class SearchResult(BaseModel):
    source_type: str
    source_path: str
    snippet: str
    score: float


class AskRequest(BaseModel):
    query: str
    limit: int = 10


@app.post("/index/filesystem", response_model=IndexResponse)
def index_filesystem(req: IndexFilesystemRequest):
    path = Path(req.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {path}")
    count = ingest.index_filesystem(path)
    return IndexResponse(indexed=count)


@app.post("/index/imessage", response_model=IndexResponse)
def index_imessage():
    try:
        count = ingest.index_imessage()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IndexResponse(indexed=count)


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


@app.get("/status")
def status():
    return {"total_chunks": store.count()}
