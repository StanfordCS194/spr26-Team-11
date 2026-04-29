"""
Cross-encoder re-ranking for search results.

The cross-encoder scores (query, document) pairs jointly, which is dramatically
more accurate than the bi-encoder cosine similarity used for initial retrieval
— but too slow to run over the full index. The standard pattern is two-stage:

    1. Retrieve top-N candidates with the cheap bi-encoder (fastembed cosine)
    2. Re-rank those N with the cross-encoder

Uses fastembed's TextCrossEncoder (same ONNX runtime as the embedding model,
~80MB on disk).
"""
from fastembed.rerank.cross_encoder import TextCrossEncoder

_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=_MODEL_NAME)
    return _model


def rerank(query: str, documents: list[str]) -> list[float]:
    """Score (query, document) pairs. Higher score = more relevant.

    Returns one float per document, in the same order as the input list. The
    raw scores are unbounded logits (typically -10 to +10). Use them as a sort
    key, not as a calibrated probability.
    """
    if not documents:
        return []
    return list(_get_model().rerank(query, documents))
