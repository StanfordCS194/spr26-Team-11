from pathlib import Path

DATA_DIR = Path.home() / ".atlas"
DB_DIR = DATA_DIR / "db"
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
