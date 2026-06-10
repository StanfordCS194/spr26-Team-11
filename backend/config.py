import json
import os
from pathlib import Path

DATA_DIR = Path.home() / ".atlas"
DB_DIR = DATA_DIR / "db"
# Per-chunk index (existing retrieval mode). One row per chunk.
DB_FILE = DB_DIR / "atlas.db"
# Per-document index (new retrieval mode). One row per file with a
# summary embedding + topics + document_type extracted by the LLM at
# index time. Lives in a separate SQLite file so the two modes can
# coexist on disk and we can A/B them by flipping retrieval_mode.
TAGGED_DB_FILE = DB_DIR / "atlas_tagged.db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400   # words
CHUNK_OVERLAP = 50  # words
IMESSAGE_DB = Path.home() / "Library" / "Messages" / "chat.db"

# File extensions to index
INDEXABLE_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".py", ".js", ".ts", ".html", ".csv", ".c", ".h", ".cpp", ".java", ".swift"}

# Directories to skip during filesystem walk
EXCLUDED_DIRS = {".venv", "venv", "node_modules", ".git", "__pycache__", ".mypy_cache", "dist", "build"}

# Daemon (FastAPI server)
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 8765
DAEMON_URL = f"http://{DAEMON_HOST}:{DAEMON_PORT}"
DAEMON_PID_FILE = DATA_DIR / "daemon.pid"
DAEMON_LOG_FILE = DATA_DIR / "daemon.log"

# ---------------------------------------------------------------------------
# User config (~/.atlas/config.json) — controls cloud-vs-local LLM parser.
# Edited via `cli.py config ...` commands. Daemon reads at startup; mode
# changes require a daemon restart.
# ---------------------------------------------------------------------------
USER_CONFIG_FILE = DATA_DIR / "config.json"

_DEFAULT_USER_CONFIG = {
    "cloud_parser": False,
    "cloud_endpoint": "https://api.groq.com/openai/v1",
    "cloud_model": "llama-3.1-8b-instant",
    # Retrieval mode: "chunk" (existing — embed every chunk, retrieve+rerank)
    # or "document" (new — one LLM-generated summary embedding per file).
    # Both indices live side by side; flip and restart the daemon to switch.
    "retrieval_mode": "chunk",
}


def load_user_config() -> dict:
    """Load ~/.atlas/config.json, filling in defaults for missing keys.

    Env var ATLAS_CLOUD_PARSER (0/1) overrides the file's cloud_parser
    setting — useful for ad-hoc testing without editing the file.
    """
    config = dict(_DEFAULT_USER_CONFIG)
    if USER_CONFIG_FILE.exists():
        try:
            config.update(json.loads(USER_CONFIG_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    env_override = os.environ.get("ATLAS_CLOUD_PARSER")
    if env_override is not None:
        config["cloud_parser"] = env_override.strip() in ("1", "true", "True", "yes")
    return config


def save_user_config(config: dict) -> None:
    """Persist a config dict to ~/.atlas/config.json, creating parent dirs."""
    USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")


# Keychain storage location for the cloud-parser API key. Stored via the
# `keyring` library; on macOS this lands in the user's login keychain.
KEYCHAIN_SERVICE = "atlas"
KEYCHAIN_USERNAME = "cloud_api_key"

# Google Calendar OAuth: client credentials JSON downloaded from Google Cloud
# Console (Desktop app type) lives next to this file. Tokens (access +
# refresh) are stored in the Keychain as a JSON blob under a separate slot.
GCAL_CLIENT_FILE = Path(__file__).parent / "gcal_client.json"
GCAL_KEYCHAIN_USERNAME = "gcal_token"
GCAL_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
]

# Gmail OAuth: separate Desktop-app client JSON and Keychain slot from Calendar.
GMAIL_CLIENT_FILE = Path(__file__).parent / "gmail_client.json"
GMAIL_KEYCHAIN_USERNAME = "gmail_token"
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]
