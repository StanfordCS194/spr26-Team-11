# -*- coding: utf-8 -*-
"""Google Calendar source for Atlas.

OAuth: InstalledAppFlow loopback redirect. Tokens persisted to the macOS
Keychain via `keyring` (matching the cloud-parser API key pattern).

Public surface:
  - authorize(interactive=True) -> Credentials | None
    Loads cached tokens from Keychain, refreshes if needed, or runs the
    full browser flow when no usable token exists. Returns None on cancel /
    decline so callers can bail cleanly instead of catching exceptions.

  - fetch_events(creds, time_min, time_max) -> list[dict]
    Calls calendarList.list() then events.list() per visible calendar.
    Returns flat list of event dicts with an injected "_calendar_name" field.

  - event_to_chunks(event) -> list[tuple[str, str, str]]
    Produces (chunk_id_suffix, embeddable_text, snippet_text) tuples with
    the [Calendar: ... | When: ... | With: ...] prefix pattern.

  - index_gcal(progress_callback=None) -> int
    Top-level entry: auth -> fetch -> chunk -> embed -> store.
    Returns the number of events indexed.

  - clear_token() -> bool
    Wipe stored credentials from the Keychain.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GCAL_CLIENT_FILE,
    GCAL_KEYCHAIN_USERNAME,
    GCAL_SCOPES,
    KEYCHAIN_SERVICE,
)

log = logging.getLogger("atlas.gcal")


# ---------------------------------------------------------------------------
# Token storage (Keychain)
# ---------------------------------------------------------------------------

def _load_token() -> "Credentials | None":
    """Return cached Credentials, or None if none stored."""
    import keyring
    from google.oauth2.credentials import Credentials

    blob = keyring.get_password(KEYCHAIN_SERVICE, GCAL_KEYCHAIN_USERNAME)
    if not blob:
        return None
    try:
        info = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return Credentials.from_authorized_user_info(info, scopes=GCAL_SCOPES)


def _save_token(creds: "Credentials") -> None:
    import keyring
    # to_json() returns a JSON string with access_token, refresh_token,
    # token_uri, client_id, client_secret, scopes, expiry.
    keyring.set_password(KEYCHAIN_SERVICE, GCAL_KEYCHAIN_USERNAME, creds.to_json())


def clear_token() -> bool:
    """Delete the stored token. Returns True if a token existed."""
    import keyring
    try:
        keyring.delete_password(KEYCHAIN_SERVICE, GCAL_KEYCHAIN_USERNAME)
        return True
    except keyring.errors.PasswordDeleteError:
        return False


def has_token() -> bool:
    import keyring
    return keyring.get_password(KEYCHAIN_SERVICE, GCAL_KEYCHAIN_USERNAME) is not None


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------

def authorize(interactive: bool = True) -> "Credentials | None":
    """Return valid Credentials, running the browser flow if necessary.

    Returns None when:
      - No token cached and `interactive=False`
      - User cancels / declines in the browser
      - Browser is closed before completion
      - Refresh fails and `interactive=False`
    Raises FileNotFoundError if the OAuth client JSON is missing.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = _load_token()

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except RefreshError as e:
            log.warning("token refresh failed (%s); clearing and re-auth needed", e)
            clear_token()
            creds = None

    if not interactive:
        return None

    if not GCAL_CLIENT_FILE.exists():
        raise FileNotFoundError(
            f"Google Calendar OAuth client JSON missing at {GCAL_CLIENT_FILE}.\n"
            "Download it from Google Cloud Console -> Credentials and save it there."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(GCAL_CLIENT_FILE), scopes=GCAL_SCOPES
        )
        # port=0 -> pick a free local port. open_browser=True is the default.
        creds = flow.run_local_server(port=0)
    except KeyboardInterrupt:
        log.info("authorization cancelled by user (KeyboardInterrupt)")
        return None
    except Exception as e:
        # google-auth-oauthlib raises various subclasses for access_denied,
        # invalid_grant, etc. We treat any failure here as "user declined"
        # since the next step (browser flow) won't have happened.
        log.warning("OAuth flow failed: %s", e)
        return None

    if creds is None:
        return None

    _save_token(creds)
    return creds


# ---------------------------------------------------------------------------
# Fetch events
# ---------------------------------------------------------------------------

# Default time window. Past is where semantic search shines; future is bounded
# to keep the index lean. Calendar.app already wins for far-future date queries.
_PAST_YEARS = 2
_FUTURE_MONTHS = 6


def _default_window() -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=365 * _PAST_YEARS)).isoformat()
    time_max = (now + timedelta(days=30 * _FUTURE_MONTHS)).isoformat()
    return time_min, time_max


def fetch_events(
    creds: "Credentials",
    time_min: str | None = None,
    time_max: str | None = None,
) -> list[dict]:
    """Fetch events from all *selected* calendars in the user's calendarList.

    "Selected" means the calendar's visibility checkbox is on in Google
    Calendar's UI — so US Holidays / Birthdays drop out automatically if the
    user hides them, no picker UI needed on our side.
    """
    from googleapiclient.discovery import build

    if time_min is None or time_max is None:
        tm, tx = _default_window()
        time_min = time_min or tm
        time_max = time_max or tx

    service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    # calendarList.list returns the user's subscribed calendars + their
    # visibility/colour metadata.
    cal_list: list[dict] = []
    page_token = None
    while True:
        resp = service.calendarList().list(pageToken=page_token).execute()
        cal_list.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    selected = [c for c in cal_list if c.get("selected", True)]

    events: list[dict] = []
    for cal in selected:
        cal_id = cal["id"]
        cal_name = cal.get("summaryOverride") or cal.get("summary") or cal_id
        page_token = None
        while True:
            resp = service.events().list(
                calendarId=cal_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
                maxResults=2500,
                pageToken=page_token,
            ).execute()
            for ev in resp.get("items", []):
                # Skip cancelled / tombstone events
                if ev.get("status") == "cancelled":
                    continue
                ev["_calendar_id"] = cal_id
                ev["_calendar_name"] = cal_name
                events.append(ev)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    return events


# ---------------------------------------------------------------------------
# Chunking — structured prefix mirrors the [Location: ...] trick in ingest.py
# ---------------------------------------------------------------------------

def _format_attendees(event: dict) -> str:
    attendees = event.get("attendees") or []
    names: list[str] = []
    for a in attendees:
        if a.get("self"):
            continue  # don't list "me" as someone I'm meeting with
        name = a.get("displayName")
        if not name:
            email = a.get("email", "")
            name = email.split("@", 1)[0] if email else ""
        if name:
            names.append(name)
    return ", ".join(names)


def _format_when(event: dict) -> str:
    """Return a YYYY-MM-DD date string for the event's start. Handles both
    timed events (dateTime) and all-day events (date)."""
    start = event.get("start") or {}
    if "dateTime" in start:
        # ISO 8601 with timezone; just slice to date for the prefix
        return start["dateTime"][:10]
    if "date" in start:
        return start["date"]
    return ""


def _chunk_text_with_overlap(text: str) -> list[str]:
    """Word-based chunker matching ingest.chunk_text() so long event
    descriptions split the same way pasted-in meeting notes would."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 0:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def event_to_chunks(event: dict) -> list[tuple[int, str, str]]:
    """Convert an event into one or more chunks ready for embedding.

    Returns a list of (chunk_index, embeddable_text, snippet_text):
      - embeddable_text has the structured prefix so the bi-encoder picks up
        calendar/date/attendee context (mirrors [Location: ...] for files).
      - snippet_text is the plain title + description (no prefix), used as
        the stored document — keeps result snippets human-readable.
    """
    title = (event.get("summary") or "(no title)").strip()
    description = (event.get("description") or "").strip()
    location = (event.get("location") or "").strip()

    cal_name = event.get("_calendar_name", "")
    when = _format_when(event)
    attendees = _format_attendees(event)

    prefix_parts = []
    if cal_name:
        prefix_parts.append(f"Calendar: {cal_name}")
    if when:
        prefix_parts.append(f"When: {when}")
    if attendees:
        prefix_parts.append(f"With: {attendees}")
    if location:
        prefix_parts.append(f"Where: {location}")
    prefix = f"[{' | '.join(prefix_parts)}]" if prefix_parts else ""

    body = title
    if description:
        body = f"{title}\n\n{description}"

    # Short events (the common case): one chunk total.
    if len(body.split()) <= CHUNK_SIZE:
        embeddable = f"{prefix}\n{body}" if prefix else body
        return [(0, embeddable, body)]

    # Long description: chunk the body, repeat the prefix on each chunk.
    out: list[tuple[int, str, str]] = []
    for i, piece in enumerate(_chunk_text_with_overlap(body)):
        embeddable = f"{prefix}\n{piece}" if prefix else piece
        out.append((i, embeddable, piece))
    return out


# ---------------------------------------------------------------------------
# Index entry point — auth -> fetch -> chunk -> embed -> store
# ---------------------------------------------------------------------------

def _event_source_path(event: dict) -> str:
    """Stable identifier used for dedup. Recurring instances collapse to the
    series so the weekly standup doesn't fill the result list."""
    series = event.get("recurringEventId") or event.get("id")
    cal = event.get("_calendar_id", "")
    return f"gcal://{cal}/{series}"


def _event_timestamp(event: dict) -> str:
    start = event.get("start") or {}
    if "dateTime" in start:
        return start["dateTime"]
    if "date" in start:
        return start["date"]
    return ""


def index_gcal(progress_callback=None) -> int:
    """Auth, fetch, chunk, embed, and store all events from all selected
    calendars. Returns the number of events indexed (not chunks)."""
    import hashlib

    import embed as embedder
    import store

    creds = authorize(interactive=True)
    if creds is None:
        log.info("authorization unavailable; skipping gcal index")
        return 0

    events = fetch_events(creds)
    total = len(events)
    if progress_callback is not None:
        progress_callback(0, total)

    indexed = 0
    for ev in events:
        chunks = event_to_chunks(ev)
        if not chunks:
            continue

        source_path = _event_source_path(ev)
        ts = _event_timestamp(ev)

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

    return indexed
