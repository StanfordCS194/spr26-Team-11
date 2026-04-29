from fastembed import TextEmbedding
from config import EMBED_MODEL

_model = None

def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    return [e.tolist() for e in _get_model().embed(texts)]

def embed_one(text: str) -> list[float]:
    return next(_get_model().embed([text])).tolist()
