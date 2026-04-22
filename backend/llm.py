# -*- coding: utf-8 -*-
"""
Local LLM for query parsing.
Auto-detects backend: mlx-lm on Apple Silicon, llama-cpp-python otherwise.
"""
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
_MODEL_ID_MLX = "mlx-community/Phi-3-mini-4k-instruct-4bit"
_MODEL_ID_GGUF_REPO = "bartowski/Phi-3-mini-4k-instruct-GGUF"
_MODEL_ID_GGUF_FILE = "Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Phi-3 chat template
_PROMPT_TEMPLATE = """\
<|system|>
You are a search query parser. Given a user's input, extract the core search terms and intent.
Output ONLY a single JSON object — no explanation, no markdown.

Keys:
- search_terms: string of core search terms, stripped of conversational words (please, find me, show me, can you, where is, what is, look for)
- intent: one of "search" (find relevant content), "find_file" (locate specific files), "find_directory" (locate a folder)
- source_filter: one of "filesystem", "imessage", or null (search all sources)

Examples:
Input: please find the directory containing my cs107 homework
Output: {{"search_terms": "cs107 homework", "intent": "find_directory", "source_filter": null}}

Input: what files do I have about machine learning
Output: {{"search_terms": "machine learning", "intent": "find_file", "source_filter": "filesystem"}}

Input: find messages from Alex about the meeting
Output: {{"search_terms": "Alex meeting", "intent": "search", "source_filter": "imessage"}}

Input: dynamic memory allocation
Output: {{"search_terms": "dynamic memory allocation", "intent": "search", "source_filter": null}}
<|end|>
<|user|>
Input: {query}
Output: \
"""


@dataclass
class ParsedQuery:
    search_terms: str
    intent: str          # "search" | "find_file" | "find_directory"
    source_filter: str | None  # "filesystem" | "imessage" | None


# ---------------------------------------------------------------------------
# Backend detection and lazy model loading
# ---------------------------------------------------------------------------

_backend: str | None = None
_model = None
_tokenizer = None


def _detect_backend() -> str:
    if platform.machine() == "arm64":
        try:
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    try:
        import llama_cpp  # noqa: F401
        return "llamacpp"
    except ImportError:
        raise RuntimeError(
            "No LLM backend found.\n"
            "  Apple Silicon: pip install mlx-lm\n"
            "  Other:         pip install llama-cpp-python"
        )


def _load_model():
    global _backend, _model, _tokenizer
    if _model is not None:
        return

    _backend = _detect_backend()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if _backend == "mlx":
        from mlx_lm import load
        print(f"Loading Phi-3-mini via MLX (first run downloads ~2.3 GB to ~/.atlas/models/)...")
        _model, _tokenizer = load(_MODEL_ID_MLX)
        print("Model ready.")
    else:
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        model_path = MODEL_DIR / _MODEL_ID_GGUF_FILE
        if not model_path.exists():
            print("Downloading Phi-3-mini GGUF (~2.3 GB, one-time)...")
            hf_hub_download(
                repo_id=_MODEL_ID_GGUF_REPO,
                filename=_MODEL_ID_GGUF_FILE,
                local_dir=str(MODEL_DIR),
            )
            print("Download complete.")
        _model = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_gpu_layers=-1,  # use all GPU layers available
            verbose=False,
        )


def _generate(prompt: str) -> str:
    _load_model()
    if _backend == "mlx":
        from mlx_lm import generate
        return generate(_model, _tokenizer, prompt=prompt, max_tokens=120, verbose=False)
    else:
        result = _model(prompt, max_tokens=120, stop=["<|end|>", "<|user|>", "\n\n"], echo=False)
        return result["choices"][0]["text"]


# ---------------------------------------------------------------------------
# JSON extraction and query parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def parse_query(text: str) -> ParsedQuery:
    """Parse a natural language query into structured search parameters."""
    prompt = _PROMPT_TEMPLATE.format(query=text.strip())
    output = _generate(prompt)
    data = _extract_json(output)

    if data is None:
        # Fallback: treat full input as search terms
        return ParsedQuery(search_terms=text, intent="search", source_filter=None)

    intent = data.get("intent", "search")
    if intent not in ("search", "find_file", "find_directory"):
        intent = "search"

    source = data.get("source_filter")
    if source not in ("filesystem", "imessage", None):
        source = None

    return ParsedQuery(
        search_terms=data.get("search_terms", text) or text,
        intent=intent,
        source_filter=source,
    )
