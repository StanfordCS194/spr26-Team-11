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
    'source_filter (one of "filesystem", "imessage", "gcal", or null). '
    'Use "gcal" for queries about calendar events, meetings, syncs, 1:1s, '
    'or anything scheduled with attendees.'
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
    source_filter: Literal["filesystem", "imessage", "gcal"] | None = None


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
# Local backend — Qwen2.5-0.5B via mlx-lm, JSON-constrained via outlines.
# Lazy-loaded on first parse_query() call.
# ---------------------------------------------------------------------------

_local_generator = None  # outlines.Generator instance


def _load_local() -> None:
    global _local_generator
    if _local_generator is not None:
        return
    if platform.machine() != "arm64":
        raise RuntimeError(
            "Local LLM parser requires Apple Silicon (mlx-lm). "
            "On other platforms, enable cloud mode: cli.py config set-cloud-parser true"
        )
    import outlines
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(_QWEN_MODEL_ID)
    mlx_model = outlines.from_mlxlm(model, tokenizer)
    _local_generator = outlines.Generator(mlx_model, output_type=_ParsedSchema)


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
