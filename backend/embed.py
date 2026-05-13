from fastembed import TextEmbedding
from config import EMBED_MODEL

_model = None

def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    # Strip unpaired surrogates and other invalid UTF-8 before the tokenizer
    # sees them — fastembed raises "TextEncodeInput must be ..." otherwise.
    safe = [t.encode("utf-8", "replace").decode("utf-8") for t in texts]
    return [e.tolist() for e in _get_model().embed(safe)]

def embed_one(text: str) -> list[float]:
    return next(_get_model().embed([text])).tolist()
