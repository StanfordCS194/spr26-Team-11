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
import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder

_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"
_model: TextCrossEncoder | None = None
_call_count = 0

# Cap doc length before reranking. The MiniLM cross-encoder tokenizes and
# truncates internally at 512 tokens (~2000 chars), so 1500 chars never loses
# ranking signal — but capping upstream stabilizes the batch shape the model
# sees. ONNX Runtime uses an arena allocator that creates a fresh memory
# region for every novel input shape and never releases it; with variable-
# length docs in each batch, that arena grew ~110 MB per request in
# production. With a uniform cap, the arena stops growing after warmup.
_MAX_DOC_CHARS = 1500

# Recycle the cross-encoder every N calls to bound arena growth. The
# truncation above stabilizes doc length, but the batch-size dimension
# still varies (dedup leaves anywhere from ~30 to ~50 items per call), so
# ONNX still allocates a new ~110 MB arena occasionally. Recycling drops
# all accumulated arenas at once. Amortized cost: a single ~100–200 ms
# reload every N calls, so per-call overhead is ~200/N ms.
# 50 caps the working set at ~2 GB of arena growth between cycles —
# enough headroom for normal usage while keeping recycle frequency low.
_RECYCLE_EVERY = 20

log = logging.getLogger("atlas.daemon")


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
    global _call_count, _model
    truncated = [d[:_MAX_DOC_CHARS] for d in documents]
    scores = list(_get_model().rerank(query, truncated))
    _call_count += 1
    if _call_count >= _RECYCLE_EVERY:
        # Drop the model so the next call lazy-reloads it. This releases
        # the ONNX inference session and all its accumulated arena
        # regions; the OS reclaims the swap/compressed pages they pinned.
        log.info("rerank: recycling cross-encoder after %d calls", _call_count)
        _model = None
        _call_count = 0
    # Force a full GC pass to release tensor wrappers and Python-side
    # intermediates that pin ONNX Runtime IO buffers. Adds ~5–10 ms per
    # rerank call; negligible compared to the ~700 ms rerank itself.
    gc.collect()
    return scores
