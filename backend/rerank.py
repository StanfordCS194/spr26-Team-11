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

import psutil
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

# Memory-based recycle: drop the cross-encoder and reload it once our
# process has grown this many MB above the baseline measured after warmup.
# The signal we use is psutil's USS (Unique Set Size) — memory unique to
# this process, including pages macOS has compressed or swapped. USS
# tracks ONNX arena growth tightly because each new MALLOC_LARGE region
# adds to it whether the pages are resident or in the compressor.
#
# A call-count trigger was the previous strategy but the leak rate
# varies per call (0–1 new arena region depending on batch-size dedup),
# so a memory-based bound is both more accurate and lets us state a hard
# upper limit on the daemon's footprint.
_MAX_GROWTH_MB = 1000

# Hard recycle floor regardless of measured growth — protects against
# psutil reading errors or unusual memory accounting where USS doesn't
# track arena allocations as expected. ~500 calls between forced
# recycles is rare for normal usage.
_HARD_CALL_LIMIT = 500

_proc = psutil.Process()  # defaults to self; works inside the daemon process
_baseline_uss: int | None = None
_call_count = 0

log = logging.getLogger("atlas.daemon")


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=_MODEL_NAME)
    return _model


def _current_uss_bytes() -> int | None:
    """Read this process's Unique Set Size in bytes. Returns None if psutil
    can't read it for any reason — the rerank path stays alive either way."""
    try:
        return _proc.memory_full_info().uss
    except Exception:
        return None


def _maybe_recycle() -> None:
    """Decide whether to drop _model based on memory growth and call count.
    Called at the end of every rerank(). Updates _baseline_uss on the first
    successful reading after a (re)load."""
    global _model, _baseline_uss, _call_count

    uss = _current_uss_bytes()
    growth_mb: int | None = None
    if uss is not None:
        if _baseline_uss is None:
            # First reading after start or after the most recent recycle —
            # use it as the baseline for subsequent comparisons.
            _baseline_uss = uss
        else:
            growth_mb = (uss - _baseline_uss) // (1024 * 1024)

    over_memory = growth_mb is not None and growth_mb >= _MAX_GROWTH_MB
    over_calls = _call_count >= _HARD_CALL_LIMIT

    if over_memory or over_calls:
        reason = (
            f"USS grew {growth_mb} MB above baseline {_baseline_uss // (1024 * 1024)} MB"
            if over_memory
            else f"hit hard call-count floor ({_call_count} calls)"
        )
        log.info("rerank: recycling cross-encoder — %s", reason)
        _model = None
        # Reset both signals — the next rerank reloads the model and the
        # baseline will be re-captured on that call's measurement.
        _baseline_uss = None
        _call_count = 0


def rerank(query: str, documents: list[str]) -> list[float]:
    """Score (query, document) pairs. Higher score = more relevant.

    Returns one float per document, in the same order as the input list. The
    raw scores are unbounded logits (typically -10 to +10). Use them as a sort
    key, not as a calibrated probability.
    """
    if not documents:
        return []
    global _call_count
    truncated = [d[:_MAX_DOC_CHARS] for d in documents]
    scores = list(_get_model().rerank(query, truncated))
    _call_count += 1
    # Force a full GC pass to release tensor wrappers and Python-side
    # intermediates that pin ONNX Runtime IO buffers. Adds ~5–10 ms per
    # rerank call; negligible compared to the ~700 ms rerank itself.
    gc.collect()
    # Check thresholds AFTER gc so the USS reading reflects the freed
    # Python-side allocations and we don't bounce on transient peaks.
    _maybe_recycle()
    return scores
