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
import gc

from fastembed.rerank.cross_encoder import TextCrossEncoder

_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_model: TextCrossEncoder | None = None

# Cap doc length before reranking. The MiniLM cross-encoder tokenizes and
# truncates internally at 512 tokens (~2000 chars), so 1500 chars never loses
# ranking signal — but capping upstream stabilizes the batch shape the model
# sees. ONNX Runtime uses an arena allocator that creates a fresh memory
# region for every novel input shape and never releases it; with variable-
# length docs in each batch, that arena grew ~110 MB per request in
# production. With a uniform cap, the arena stops growing after warmup.
_MAX_DOC_CHARS = 1500


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
    truncated = [d[:_MAX_DOC_CHARS] for d in documents]
    scores = list(_get_model().rerank(query, truncated))
    # Force a full GC pass to release tensor wrappers and Python-side
    # intermediates that pin ONNX Runtime IO buffers. Adds ~5–10 ms per
    # rerank call; negligible compared to the ~700 ms rerank itself.
    gc.collect()
    return scores
