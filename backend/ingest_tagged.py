# -*- coding: utf-8 -*-
"""
Document-mode indexer.

For each indexable file:
  1. Parse to text (reusing ingest._parse_file)
  2. Sample a stitched excerpt across the file (tagging.sample_file_text)
  3. Ask the LLM for {topics, document_type, summary} (tagging.extract_document_tags)
  4. Embed the LLM-generated summary (embed.embed_one)
  5. Upsert one row into the tagged store

Failure policy: if any step raises for a single file, log it and skip
that file. One bad file shouldn't break the whole index run.

Speed envelope (per the plan): roughly 2 s/file on local Qwen-0.5B, 0.5–1
s/file on cloud Groq llama-3.1-8b. For 2,500 files that's ~80 min local,
~30 min cloud.
"""
import logging
import time
from pathlib import Path

from config import INDEXABLE_EXTENSIONS, EXCLUDED_DIRS
import embed as embedder
import store_tagged
import tagging
from ingest import _parse_file, _path_to_context

log = logging.getLogger("atlas.daemon")

# Retry policy for transient LLM failures (rate limits, network blips,
# malformed JSON). After this many attempts a file is logged + skipped.
_MAX_RETRIES = 3
_INITIAL_BACKOFF_SEC = 5.0

# Proactive pacing for the cloud LLM path. Groq's free tier has two
# overlapping limits: 30 RPM and ~6,000 TPM (tokens per minute) for
# llama-3.1-8b-instant. Tag extraction averages ~1,100 tokens per call
# (system + sampled file excerpt + 300-token output), so the TPM cap
# binds first: 6000 / 1100 ≈ 5.5 RPM => one call every ~11 s. We pace at
# 8 s with a small safety margin; running slightly under TPM avoids the
# 429 storms (each lost ~30+ s to retry backoff) and keeps the run
# steady. No-op for local mode (no rate limit).
_CLOUD_MIN_INTERVAL_SEC = 8.0


def index_filesystem(
    root: Path,
    verbose: bool = False,
    progress_callback=None,
) -> int:
    """Walk `root`, tag each file via the LLM, store summary embeddings.
    Returns the number of files successfully indexed."""
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in INDEXABLE_EXTENSIONS
        and not any(part in EXCLUDED_DIRS for part in p.parts)
    ]
    total = len(files)
    indexed = 0
    failed = 0

    # Detect cloud mode once so we can apply pacing without re-loading
    # config every iteration.
    import llm as _llm
    cloud_mode = _llm.mode() == "cloud"
    last_llm_call_ts = 0.0

    for path in files:
        try:
            text = _parse_file(path)
            if not text or not text.strip():
                continue

            context_label, path_tokens_str = _path_to_context(path)

            # Retry the LLM call with exponential backoff on transient
            # errors. The two dominant failures with Groq are:
            #   - HTTP 429 (rate limit). Groq's 429 response carries a
            #     Retry-After header telling us exactly how long to sleep;
            #     we honor it instead of guessing with exponential backoff.
            #   - Schema validation errors (the model occasionally returns
            #     summary as a list of strings or other variant). The
            #     normalizer in llm.py handles the most common variants;
            #     anything left is a hard schema mismatch that retrying
            #     won't fix, so we don't retry validation errors.
            tags = None
            backoff = _INITIAL_BACKOFF_SEC
            for attempt in range(_MAX_RETRIES):
                try:
                    # Proactive cloud-mode pacing — stay under 30 RPM.
                    if cloud_mode:
                        elapsed = time.time() - last_llm_call_ts
                        if elapsed < _CLOUD_MIN_INTERVAL_SEC:
                            time.sleep(_CLOUD_MIN_INTERVAL_SEC - elapsed)
                    tags = tagging.extract_document_tags(
                        context_label or str(path), text,
                    )
                    last_llm_call_ts = time.time()
                    break
                except Exception as e:
                    # Don't retry deterministic failures.
                    is_validation_error = "validation error" in str(e).lower()
                    if is_validation_error or attempt == _MAX_RETRIES - 1:
                        raise

                    # Prefer Groq's Retry-After header if the underlying
                    # exception is an HTTP error carrying it.
                    sleep_for = backoff
                    resp = getattr(e, "response", None)
                    if resp is not None:
                        retry_after = resp.headers.get("Retry-After") if resp.headers else None
                        if retry_after:
                            try:
                                sleep_for = max(float(retry_after), 1.0)
                            except (TypeError, ValueError):
                                pass

                    log.info(
                        "tagged index: retrying %s after %.1fs (attempt %d/%d): %s",
                        path.name, sleep_for, attempt + 1, _MAX_RETRIES, e,
                    )
                    time.sleep(sleep_for)
                    backoff *= 2

            # The embedding target is "topics + summary" stitched together.
            # Topics alone are too sparse for cosine; the summary alone
            # misses the explicit topic tokens (cs107, slab, etc.) that the
            # LLM thought important. Concatenating gives both signals one
            # vector representation.
            embed_text = ""
            if tags.topics:
                embed_text = ", ".join(tags.topics)
            if tags.summary:
                embed_text = f"{embed_text}. {tags.summary}" if embed_text else tags.summary
            if not embed_text:
                # If the LLM produced nothing usable, fall back to the path
                # label so the file is at least retrievable by name.
                embed_text = context_label or str(path)

            summary_embedding = embedder.embed_one(embed_text)

            store_tagged.add_document(
                source_path=str(path),
                source_type="filesystem",
                document_type=tags.document_type,
                topics=tags.topics,
                path_tokens=path_tokens_str,
                summary=tags.summary,
                summary_embedding=summary_embedding,
            )
            indexed += 1

        except Exception as e:
            failed += 1
            log.warning("tagged index: skipping %s (%s)", path, e)

        if progress_callback is not None:
            progress_callback(indexed, total)
        if verbose:
            print(f"  tagged: {path.relative_to(root)}")

    if failed:
        log.info("tagged index complete: %d indexed, %d failed", indexed, failed)
    return indexed


def index_imessage(verbose: bool = False) -> int:
    """iMessage data doesn't fit the doc-summary model (no per-file
    boundary, no useful 2-sentence summary per conversation). Document
    mode is filesystem-only for now — flip retrieval_mode back to "chunk"
    in ~/.atlas/config.json if you want iMessage indexing."""
    raise NotImplementedError(
        "iMessage indexing is not supported in document retrieval mode. "
        "Switch retrieval_mode to 'chunk' to index iMessage."
    )


# ---------------------------------------------------------------------------
# Google Calendar ingestion (document mode). Each event becomes one row in
# the documents table. We deliberately do NOT run the LLM tagging pass —
# event titles + descriptions are already short and high-signal, and a
# per-event LLM call would slow indexing by ~8s/event (cloud-paced) for
# negligible retrieval gain.
# ---------------------------------------------------------------------------

def index_gcal(progress_callback=None, verbose: bool = False) -> int:
    """Authorize, fetch, and store one document row per gcal event.
    Returns the number of events indexed."""
    import gcal

    creds = gcal.authorize(interactive=True)
    if creds is None:
        return 0

    events = gcal.fetch_events(creds)
    total = len(events)
    if progress_callback is not None:
        progress_callback(0, total)

    indexed = 0
    for ev in events:
        chunks = gcal.event_to_chunks(ev)
        if not chunks:
            continue

        source_path = gcal._event_source_path(ev)

        # Most events fit in one chunk; long descriptions might split. Join
        # both the embeddable text (full prefix + body, what the bi-encoder
        # sees) and the snippet text (UI-facing) so the document row carries
        # the whole event regardless of chunk count.
        embeddable_text = "\n".join(t for _, t, _ in chunks)
        snippet_text = "\n".join(s for _, _, s in chunks)

        summary_embedding = embedder.embed_one(embeddable_text)

        store_tagged.add_document(
            source_path=source_path,
            source_type="gcal",
            document_type="calendar event",
            topics=[],
            path_tokens="",
            summary=snippet_text,
            summary_embedding=summary_embedding,
        )

        indexed += 1
        if progress_callback is not None:
            progress_callback(indexed, total)
        if verbose:
            print(f"  tagged event: {source_path}")

    return indexed
