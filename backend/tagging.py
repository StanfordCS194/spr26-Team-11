# -*- coding: utf-8 -*-
"""
Document-mode indexing helpers.

This module sits between `ingest_tagged.py` (which walks the filesystem)
and `llm.py` (which produces the structured tag output). Its job is to
turn a parsed file's full text into a compact, representative excerpt that
fits inside the LLM's context window, then call out for the tags.

Why stitched samples vs first-N-chars: a 100-page PDF's first 2,000 chars
are usually a title page + table of contents — terrible signal. Stitching
head + middle + tail samples gives the LLM a much better chance of
identifying topics that only appear in the body.
"""
from dataclasses import dataclass

import llm


# ---------------------------------------------------------------------------
# Sampling. Total budget chosen to leave headroom for the LLM's prompt
# overhead (~500 tokens) inside Qwen-0.5B's 4k context. 2400 chars ≈ 600
# tokens, conservative.
# ---------------------------------------------------------------------------

_TOTAL_BUDGET_CHARS = 2400
_NUM_SAMPLES = 3  # head, middle, tail


def sample_file_text(text: str) -> str:
    """Stitch together up to _NUM_SAMPLES windows from across the file.

    For files shorter than _TOTAL_BUDGET_CHARS, returns the full text. For
    longer files, takes equally-spaced windows of size budget // num_samples
    each and joins them with a visible separator so the LLM understands
    they're non-contiguous.
    """
    text = (text or "").strip()
    if len(text) <= _TOTAL_BUDGET_CHARS:
        return text

    window = _TOTAL_BUDGET_CHARS // _NUM_SAMPLES
    # Anchor the windows at evenly-spaced positions across the file. For
    # 3 samples and file length L, anchors are at L*0/N, L*1/N, L*(N-1)/N
    # which is equivalent to head, ~middle, tail. We deliberately overshoot
    # slightly toward the head to avoid the (usually-noisy) very-last bytes
    # of binary-leaking text.
    positions = [
        int(len(text) * i / _NUM_SAMPLES)
        for i in range(_NUM_SAMPLES)
    ]
    chunks = [text[p:p + window] for p in positions]
    return "\n\n[...]\n\n".join(chunks)


# ---------------------------------------------------------------------------
# Orchestration: sample → LLM call → normalize. Indexer calls this once per
# file.
# ---------------------------------------------------------------------------

@dataclass
class DocumentTags:
    topics: list[str]
    document_type: str
    summary: str


def extract_document_tags(file_label: str, file_text: str) -> DocumentTags:
    """Sample the file's text, hand it to the LLM, return structured tags.

    `file_label` is a short cleaned path label (e.g. "Stanford CS 107
    hw3_solutions") — supplied to the LLM as a hint so it can disambiguate
    files whose contents alone might not reveal the topic.

    Raises whatever llm.extract_tags raises — caller handles retries /
    skip-and-continue policy.
    """
    sampled = sample_file_text(file_text)
    raw = llm.extract_tags(file_label, sampled)
    # Normalize: strip whitespace, drop empty topics, cap topic count.
    topics = [t.strip() for t in raw.get("topics", []) if t and t.strip()]
    return DocumentTags(
        topics=topics[:8],
        document_type=(raw.get("document_type") or "").strip(),
        summary=(raw.get("summary") or "").strip(),
    )
