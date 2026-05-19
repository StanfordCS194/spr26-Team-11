import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import store
import embed as embedder
from config import CHUNK_SIZE, CHUNK_OVERLAP, INDEXABLE_EXTENSIONS, EXCLUDED_DIRS, IMESSAGE_DB


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _chunk_id(source_type: str, source_path: str, chunk_index: int) -> str:
    key = f"{source_type}:{source_path}:{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Path context extraction
# ---------------------------------------------------------------------------

# Filesystem noise words that carry no semantic signal
_PATH_NOISE = {
    "users", "desktop", "downloads", "documents", "library",
    "home", "volumes", "private", "var", "folders", "tmp",
}

def _path_to_context(path: Path) -> tuple[str, str]:
    """
    Returns (context_label, path_tokens_str).
    context_label: human-readable prefix for embedding (e.g. "Stanford CS 107 Frosh Winter hw3")
    path_tokens_str: space-separated lowercase tokens for keyword scoring
    """
    try:
        rel = path.relative_to(Path.home())
        parts = list(rel.parts)
    except ValueError:
        parts = list(path.parts)

    # Include directory components + file stem (name without extension)
    components = parts[:-1] + [path.stem]

    tokens = []
    for component in components:
        # Split on any non-alphanumeric run (handles spaces, underscores, hyphens, dots)
        sub = re.split(r'[^a-zA-Z0-9]+', component)
        for t in sub:
            # Keep alphanumeric tokens > 1 char; filter single-digit ordinals (1, 2, 3...)
            # but keep meaningful numbers like course numbers (107, 194, etc.)
            if t and len(t) > 1 and t.lower() not in _PATH_NOISE and not (t.isdigit() and len(t) < 3):
                tokens.append(t)

    label = " ".join(tokens)
    path_tokens_str = " ".join(t.lower() for t in tokens)
    return label, path_tokens_str


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

def _parse_file(path: Path) -> str | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".js", ".ts", ".html", ".csv", ".c", ".h", ".cpp", ".java", ".swift", ""}:
            return path.read_text(errors="ignore")
        elif suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Filesystem ingestion
# ---------------------------------------------------------------------------

def index_filesystem(root: Path, verbose: bool = False, progress_callback=None) -> int:
    indexed = 0
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in INDEXABLE_EXTENSIONS
        and not any(part in EXCLUDED_DIRS for part in p.parts)
    ]

    for path in files:
        try:
            text = _parse_file(path)
            if not text or not text.strip():
                continue

            context_label, path_tokens_str = _path_to_context(path)
            raw_chunks = chunk_text(text)
            if not raw_chunks:
                continue

            # Prepend location context to each chunk before embedding
            chunks_for_embedding = [
                f"[Location: {context_label}]\n\n{chunk}" if context_label else chunk
                for chunk in raw_chunks
            ]

            source_path = str(path)
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

            ids = [_chunk_id("filesystem", source_path, i) for i in range(len(raw_chunks))]
            metadatas = [
                {"source_type": "filesystem", "source_path": source_path, "chunk_index": i, "timestamp": mtime}
                for i in range(len(raw_chunks))
            ]
            embeddings = embedder.embed(chunks_for_embedding)

            # Store raw chunk text (without location prefix) so snippets stay readable
            store.add(
                ids=ids,
                embeddings=embeddings,
                documents=raw_chunks,
                metadatas=metadatas,
                path_tokens=[path_tokens_str] * len(raw_chunks),
            )

            indexed += 1
            if progress_callback is not None:
                progress_callback(indexed, len(files))
            if verbose:
                print(f"  indexed: {path.relative_to(root)} ({len(raw_chunks)} chunks)")
        except Exception as e:
            # One bad file (malformed PDF, unpaired surrogates choking the
            # tokenizer, etc.) shouldn't kill the entire index run.
            if verbose:
                print(f"  skipped: {path}: {e}")
            continue

    return indexed


# ---------------------------------------------------------------------------
# iMessage ingestion
# ---------------------------------------------------------------------------

def index_imessage(verbose: bool = False) -> int:
    if not IMESSAGE_DB.exists():
        raise FileNotFoundError(
            f"iMessage database not found at {IMESSAGE_DB}.\n"
            "Grant Full Disk Access to Terminal in System Settings > Privacy & Security."
        )

    con = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
    cur = con.cursor()

    cur.execute("""
        SELECT
            h.id          AS contact,
            m.text        AS body,
            m.date        AS apple_date,
            m.is_from_me  AS is_from_me
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.rowid
        JOIN chat c               ON c.rowid = cmj.chat_id
        JOIN chat_handle_join chj ON chj.chat_id = c.rowid
        JOIN handle h             ON h.rowid = chj.handle_id
        WHERE m.text IS NOT NULL AND m.text != ''
        ORDER BY h.id, m.date
    """)
    rows = cur.fetchall()
    con.close()

    conversations: dict[str, list[tuple]] = {}
    for contact, body, apple_date, is_from_me in rows:
        conversations.setdefault(contact, []).append((body, apple_date, is_from_me))

    indexed = 0
    for contact, messages in conversations.items():
        lines = []
        for body, apple_date, is_from_me in messages:
            speaker = "Me" if is_from_me else contact
            ts = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp() + apple_date / 1e9
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"[{dt}] {speaker}: {body}")

        text = "\n".join(lines)
        chunks = chunk_text(text)
        if not chunks:
            continue

        ids = [_chunk_id("imessage", contact, i) for i in range(len(chunks))]
        metadatas = [
            {"source_type": "imessage", "source_path": contact, "chunk_index": i, "timestamp": ""}
            for i in range(len(chunks))
        ]
        embeddings = embedder.embed(chunks)
        store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            path_tokens=[""] * len(chunks),
        )

        indexed += 1
        if verbose:
            print(f"  indexed conversation with: {contact} ({len(chunks)} chunks)")

    return indexed


# ---------------------------------------------------------------------------
# Google Calendar ingestion. Gcal-specific auth, fetch, and chunking live in
# `gcal.py`; this function owns the embed + store half so the store choice
# stays consistent with the rest of the chunk-mode pipeline.
# ---------------------------------------------------------------------------

def index_gcal(progress_callback=None, verbose: bool = False) -> int:
    """Authorize, fetch, chunk, embed, and store all events from the user's
    selected calendars. Returns the number of events indexed (not chunks)."""
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
        ts = gcal._event_timestamp(ev)

        ids = [
            hashlib.md5(f"gcal:{source_path}:{ci}".encode()).hexdigest()
            for ci, _, _ in chunks
        ]
        embeddable = [text for _, text, _ in chunks]
        snippets = [snippet for _, _, snippet in chunks]
        metadatas = [
            {"source_type": "gcal", "source_path": source_path, "chunk_index": ci, "timestamp": ts}
            for ci, _, _ in chunks
        ]
        embeddings = embedder.embed(embeddable)

        store.add(
            ids=ids,
            embeddings=embeddings,
            documents=snippets,
            metadatas=metadatas,
            path_tokens=[""] * len(chunks),  # events have no paths; same as imessage
        )

        indexed += 1
        if progress_callback is not None:
            progress_callback(indexed, total)
        if verbose:
            print(f"  indexed event: {source_path} ({len(chunks)} chunks)")

    return indexed
