# -*- coding: utf-8 -*-
"""
Query parser for `/ask`. Two modes, chosen at module import time from
`config.load_user_config()`:

  - local (default): Qwen2.5-0.5B-Instruct (4-bit MLX), output constrained
    to a valid JSON object via the `outlines` library. ~400 MB resident.
  - cloud (opt-in): OpenAI-compatible chat-completions endpoint (default
    Groq) with `response_format={"type": "json_object"}`. ~0 MB resident.

Only the active mode's heavy deps are imported. Switching modes requires a
daemon restart. The API key for cloud mode is read from the macOS Keychain
via `keyring`; the CLI's `config set-cloud-parser true` flow prompts for it.
"""
import json
import os
import platform
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from config import (
    KEYCHAIN_SERVICE,
    KEYCHAIN_USERNAME,
    load_user_config,
)


# ---------------------------------------------------------------------------
# Mode + shared prompt
# ---------------------------------------------------------------------------

_config = load_user_config()
_MODE = "cloud" if _config["cloud_parser"] else "local"

_QWEN_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

_SYSTEM_PROMPT = (
    "You are a search query parser. Extract the core search terms and intent "
    "from the user's input. Output only a JSON object with three fields: "
    "search_terms (string of core terms, stripped of conversational filler), "
    'intent (one of "search", "find_file", "find_directory"), and '
    'source_filter (one of "filesystem", "imessage", "gcal", "gmail", or null). '
    'Use "gcal" for queries about calendar events, meetings, syncs, 1:1s, '
    'or anything scheduled with attendees. Use "gmail" for email messages, '
    'threads, senders, subjects, or inbox content.'
)

_FEW_SHOT = [
    ("please find the directory containing my cs107 homework",
     {"search_terms": "cs107 homework", "intent": "find_directory", "source_filter": None}),
    ("what files do I have about machine learning",
     {"search_terms": "machine learning", "intent": "find_file", "source_filter": "filesystem"}),
    ("find messages from Alex about the meeting",
     {"search_terms": "Alex meeting", "intent": "search", "source_filter": "imessage"}),
    ("dynamic memory allocation",
     {"search_terms": "dynamic memory allocation", "intent": "search", "source_filter": None}),
    ("the meeting where we discussed the redesign",
     {"search_terms": "meeting redesign", "intent": "search", "source_filter": "gcal"}),
    ("my 1:1 with Alice last month",
     {"search_terms": "1:1 Alice", "intent": "search", "source_filter": "gcal"}),
    ("the planning sync about Q2 roadmap",
     {"search_terms": "planning Q2 roadmap", "intent": "search", "source_filter": "gcal"}),
    ("when did we have the design review",
     {"search_terms": "design review", "intent": "search", "source_filter": "gcal"}),
    ("email from Alex about the budget",
     {"search_terms": "Alex budget", "intent": "search", "source_filter": "gmail"}),
    ("find the message about project kickoff",
     {"search_terms": "project kickoff", "intent": "search", "source_filter": "gmail"}),
]


def _build_messages(user_input: str) -> list[dict]:
    """Build the chat-format messages list shared by both modes."""
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for example_in, example_out in _FEW_SHOT:
        messages.append({"role": "user", "content": example_in})
        messages.append({"role": "assistant", "content": json.dumps(example_out)})
    messages.append({"role": "user", "content": user_input.strip()})
    return messages


# ---------------------------------------------------------------------------
# Pydantic schema — same shape used by outlines (local) and as a sanity
# check on cloud responses.
# ---------------------------------------------------------------------------

class _ParsedSchema(BaseModel):
    search_terms: str = Field(default="")
    intent: Literal["search", "find_file", "find_directory"] = "search"
    source_filter: Literal["filesystem", "imessage", "gcal", "gmail"] | None = None


class _TagsSchema(BaseModel):
    """LLM-extracted per-file tags. Used by the document-mode indexer.

    `summary` accepts either a string or a list of strings — Groq's
    llama-3.1-8b occasionally returns the summary as a list of sentences
    instead of a single paragraph. Normalized to a single string by
    `_normalize_tags` below.
    """
    topics: list[str] = Field(default_factory=list, max_length=8)
    document_type: str = Field(default="")
    summary: str | list[str] = Field(default="")


def _normalize_tags(parsed: "_TagsSchema") -> dict:
    """Flatten list-shaped fields to the dict shape callers expect."""
    summary = parsed.summary
    if isinstance(summary, list):
        summary = " ".join(s.strip() for s in summary if isinstance(s, str) and s.strip())
    return {
        "topics": parsed.topics,
        "document_type": parsed.document_type,
        "summary": summary,
    }


@dataclass
class ParsedQuery:
    search_terms: str
    intent: str          # "search" | "find_file" | "find_directory"
    source_filter: str | None  # "filesystem" | "imessage" | None


def _to_dataclass(schema: _ParsedSchema, fallback_terms: str) -> ParsedQuery:
    return ParsedQuery(
        search_terms=schema.search_terms or fallback_terms,
        intent=schema.intent,
        source_filter=schema.source_filter,
    )


# ---------------------------------------------------------------------------
# Local backend — Qwen2.5-0.5B via mlx-lm.
# Two generation paths share the same loaded model:
#   - parse_query (constrained JSON via outlines)
#   - synthesize_answer (free-form text via raw mlx_lm.generate)
# Lazy-loaded on first use.
# ---------------------------------------------------------------------------

_local_model = None       # raw mlx model
_local_tokenizer = None   # raw mlx tokenizer
_local_generator = None   # outlines.Generator wrapping the above for parse_query
_local_tag_generator = None  # outlines.Generator for extract_tags


def _load_local() -> None:
    global _local_model, _local_tokenizer, _local_generator, _local_tag_generator
    if _local_generator is not None:
        return
    if platform.machine() != "arm64":
        raise RuntimeError(
            "Local LLM parser requires Apple Silicon (mlx-lm). "
            "On other platforms, enable cloud mode: cli.py config set-cloud-parser true"
        )
    import outlines
    from mlx_lm import load as mlx_load

    _local_model, _local_tokenizer = mlx_load(_QWEN_MODEL_ID)
    mlx_model = outlines.from_mlxlm(_local_model, _local_tokenizer)
    _local_generator = outlines.Generator(mlx_model, output_type=_ParsedSchema)
    # A second generator constrained to the tags schema. Same underlying
    # model — outlines just swaps logits processors per generator instance.
    _local_tag_generator = outlines.Generator(mlx_model, output_type=_TagsSchema)


def _parse_local(user_input: str) -> ParsedQuery:
    _load_local()
    messages = _build_messages(user_input)
    # outlines accepts a Chat input that triggers tokenizer.apply_chat_template
    # under the hood, but we already have a list of dicts — pass it as a Chat.
    from outlines.inputs import Chat
    raw = _local_generator(Chat(messages), max_tokens=120)
    schema = _ParsedSchema.model_validate_json(raw)
    return _to_dataclass(schema, user_input)


# ---------------------------------------------------------------------------
# Cloud backend — OpenAI-compatible chat completions (Groq by default).
# No persistent state, just an HTTP request per /ask call.
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Pull the API key from the Keychain. Env var override for testing."""
    env_key = os.environ.get("ATLAS_CLOUD_API_KEY")
    if env_key:
        return env_key
    import keyring
    key = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_USERNAME)
    if not key:
        raise RuntimeError(
            "Cloud parser enabled but no API key in Keychain. "
            "Run: cli.py config set-cloud-parser true"
        )
    return key


def _parse_cloud(user_input: str) -> ParsedQuery:
    import requests

    endpoint = _config["cloud_endpoint"].rstrip("/")
    payload = {
        "model": _config["cloud_model"],
        "messages": _build_messages(user_input),
        "response_format": {"type": "json_object"},
        "max_tokens": 120,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        f"{endpoint}/chat/completions",
        json=payload,
        headers=headers,
        timeout=10.0,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    schema = _ParsedSchema.model_validate_json(content)
    return _to_dataclass(schema, user_input)


# ---------------------------------------------------------------------------
# Tag extraction — document-mode indexer feeds a sampled excerpt of each file
# through the LLM to produce {topics, document_type, summary}. Constrained
# JSON in both modes (outlines on local, response_format on cloud) so the
# indexer can trust the output shape.
# ---------------------------------------------------------------------------

_SYSTEM_TAGS = (
    "You extract metadata from a file. Given the file's path and a stitched "
    "excerpt of its contents, return a JSON object with three fields: "
    "topics (3-7 short phrases capturing what the file is about), "
    "document_type (one of: lecture notes, homework, research paper, code, "
    "personal note, email, message, reference, other), and "
    "summary (2-3 sentences describing the file's content). "
    "Base your answer only on the path and excerpt provided."
)


def _build_tag_messages(file_label: str, sampled_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_TAGS},
        {"role": "user",
         "content": f"Path: {file_label}\n\nExcerpt:\n{sampled_text}"},
    ]


def _extract_tags_local(file_label: str, sampled_text: str) -> dict:
    _load_local()
    from outlines.inputs import Chat
    raw = _local_tag_generator(
        Chat(_build_tag_messages(file_label, sampled_text)),
        max_tokens=300,
    )
    return _normalize_tags(_TagsSchema.model_validate_json(raw))


def _extract_tags_cloud(file_label: str, sampled_text: str) -> dict:
    import requests

    endpoint = _config["cloud_endpoint"].rstrip("/")
    payload = {
        "model": _config["cloud_model"],
        "messages": _build_tag_messages(file_label, sampled_text),
        "response_format": {"type": "json_object"},
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        f"{endpoint}/chat/completions",
        json=payload, headers=headers, timeout=20.0,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return _normalize_tags(_TagsSchema.model_validate_json(content))


# ---------------------------------------------------------------------------
# Free-form synthesis — answers a question from a list of retrieved chunks.
# Used by the /query RAG endpoint. Same backend mode as parse_query, but
# generation is unconstrained (no outlines, no response_format) so the model
# can produce a natural-language answer instead of a JSON object.
# ---------------------------------------------------------------------------

# Cap each chunk's contribution to the prompt so the context window stays
# comfortably within Qwen-0.5B's 4k tokens (Llama-3.1-8b has much more, but
# we keep the same shape across modes for consistency).
_RAG_CHUNK_CHARS = 800
_RAG_MAX_TOKENS = 400

_SYSTEM_RAG = (
    "You are a helpful assistant that answers the user's question using only "
    "the information from the numbered documents provided. Cite sources by "
    "referencing the document number in square brackets like [1] or [2]. If "
    "the documents don't contain enough information to answer, say so plainly "
    "rather than guessing."
)


def _build_rag_messages(question: str, contexts: list[dict]) -> list[dict]:
    """contexts: list of {'source_path': str, 'text': str} (text is the chunk
    body — truncated upstream to _RAG_CHUNK_CHARS)."""
    docs = "\n\n".join(
        f"[{i + 1}] Source: {c['source_path']}\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    return [
        {"role": "system", "content": _SYSTEM_RAG},
        {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {question}"},
    ]


def _synthesize_local(question: str, contexts: list[dict]) -> str:
    _load_local()
    from mlx_lm import generate as mlx_generate

    messages = _build_rag_messages(question, contexts)
    prompt = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(
        _local_model, _local_tokenizer,
        prompt=prompt, max_tokens=_RAG_MAX_TOKENS, verbose=False,
    )


def _synthesize_cloud(question: str, contexts: list[dict]) -> str:
    import requests

    endpoint = _config["cloud_endpoint"].rstrip("/")
    payload = {
        "model": _config["cloud_model"],
        "messages": _build_rag_messages(question, contexts),
        "max_tokens": _RAG_MAX_TOKENS,
        # Slight temperature for more natural prose; structured-output paths
        # use 0.0, but free-form answers feel stilted at temperature=0.
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    r = requests.post(
        f"{endpoint}/chat/completions",
        json=payload,
        headers=headers,
        # Synthesis can take longer than parse — give it more headroom.
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mode() -> str:
    """Return 'local' or 'cloud' — the active parser mode at startup."""
    return _MODE


def parse_query(text: str) -> ParsedQuery:
    """Parse a natural-language query into structured search parameters.

    Raises on backend failure; callers are expected to catch and fall back
    (search.ask() falls back to plain search with the raw input).
    """
    if _MODE == "cloud":
        return _parse_cloud(text)
    return _parse_local(text)


def extract_tags(file_label: str, sampled_text: str) -> dict:
    """Extract {topics, document_type, summary} from a sampled file excerpt.

    `file_label` is a short human-readable hint about the path (e.g.
    "Stanford CS 107 hw3_solutions"). Returns a dict matching _TagsSchema;
    fields may be empty strings/lists if the model decided it couldn't tell.
    Raises on backend failure — the indexer is responsible for error
    handling and skipping/retrying files.
    """
    if _MODE == "cloud":
        return _extract_tags_cloud(file_label, sampled_text)
    return _extract_tags_local(file_label, sampled_text)


def synthesize_answer(question: str, contexts: list[dict]) -> str:
    """Generate a free-form answer to `question` using the provided contexts.

    Each context is a dict with keys 'source_path' and 'text'. The caller is
    responsible for truncating each chunk's text to a reasonable length
    (see _RAG_CHUNK_CHARS as a guideline) — this function does not truncate.

    Raises on backend failure; search.query() catches and returns the
    sources alone with a graceful "couldn't synthesize" message.
    """
    if _MODE == "cloud":
        return _synthesize_cloud(question, contexts)
    return _synthesize_local(question, contexts)
