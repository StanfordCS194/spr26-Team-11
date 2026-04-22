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
