# -*- coding: utf-8 -*-
"""Gmail source for Atlas.

OAuth: same InstalledAppFlow + Keychain pattern as `gcal.py`, with a
separate client JSON and token slot so Calendar and Mail credentials do
not collide.

Public surface:
  - authorize(interactive=True) -> Credentials | None
  - fetch_messages(creds) -> list[dict]
  - message_to_chunks(message) -> list[tuple[int, str, str]]
  - _message_source_path(message) -> str
  - _message_timestamp(message) -> str
  - clear_token() / has_token()

Indexing orchestration lives in `ingest.py` and `ingest_tagged.py`.
"""
import base64
import json
import logging
import re
from datetime import datetime, timezone
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    GMAIL_CLIENT_FILE,
    GMAIL_KEYCHAIN_USERNAME,
    GMAIL_SCOPES,
    KEYCHAIN_SERVICE,
)

log = logging.getLogger("atlas.gmail")

# Conservative defaults — opt-in via `--gmail` only.
_LOOKBACK_DAYS = 180
_MAX_MESSAGES = 500
_GMAIL_QUERY = (
    f"newer_than:{_LOOKBACK_DAYS}d in:inbox -in:spam -in:trash"
)


# ---------------------------------------------------------------------------
# Token storage (Keychain)
# ---------------------------------------------------------------------------

def _load_token() -> "Credentials | None":
    import keyring
    from google.oauth2.credentials import Credentials

    blob = keyring.get_password(KEYCHAIN_SERVICE, GMAIL_KEYCHAIN_USERNAME)
    if not blob:
        return None
    try:
        info = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return Credentials.from_authorized_user_info(info, scopes=GMAIL_SCOPES)


def _save_token(creds: "Credentials") -> None:
    import keyring
    keyring.set_password(KEYCHAIN_SERVICE, GMAIL_KEYCHAIN_USERNAME, creds.to_json())


def clear_token() -> bool:
    import keyring
    try:
        keyring.delete_password(KEYCHAIN_SERVICE, GMAIL_KEYCHAIN_USERNAME)
        return True
    except keyring.errors.PasswordDeleteError:
        return False


def has_token() -> bool:
    import keyring
    return keyring.get_password(KEYCHAIN_SERVICE, GMAIL_KEYCHAIN_USERNAME) is not None


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------

def authorize(interactive: bool = True) -> "Credentials | None":
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

    if not GMAIL_CLIENT_FILE.exists():
        raise FileNotFoundError(
            f"Gmail OAuth client JSON missing at {GMAIL_CLIENT_FILE}.\n"
            "Download it from Google Cloud Console -> Credentials and save it there."
        )

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(GMAIL_CLIENT_FILE), scopes=GMAIL_SCOPES
        )
        creds = flow.run_local_server(port=0)
    except KeyboardInterrupt:
        log.info("authorization cancelled by user (KeyboardInterrupt)")
        return None
    except Exception as e:
        log.warning("OAuth flow failed: %s", e)
        return None

    if creds is None:
        return None

    _save_token(creds)
    return creds


# ---------------------------------------------------------------------------
# Fetch messages
# ---------------------------------------------------------------------------

def fetch_messages(
    creds: "Credentials",
    max_results: int = _MAX_MESSAGES,
) -> list[dict]:
    """List recent inbox messages and hydrate each into a normalized dict.

    Returns [] on API failure so the daemon never crashes when Gmail is
    unavailable or authorization was revoked.
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    except Exception as e:
        log.error("failed to build Gmail client: %s", e)
        return []

    message_ids: list[str] = []
    page_token = None
    try:
        while len(message_ids) < max_results:
            resp = (
                service.users()
                .messages()
                .list(
                    userId="me",
                    q=_GMAIL_QUERY,
                    maxResults=min(100, max_results - len(message_ids)),
                    pageToken=page_token,
                )
                .execute()
            )
            for item in resp.get("messages", []):
                mid = item.get("id")
                if mid:
                    message_ids.append(mid)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except HttpError as e:
        log.error("Gmail messages.list failed: %s", e)
        return []
    except Exception as e:
        log.error("Gmail messages.list failed: %s", e)
        return []

    messages: list[dict] = []
    for mid in message_ids[:max_results]:
        try:
            raw = (
                service.users()
                .messages()
                .get(userId="me", id=mid, format="full")
                .execute()
            )
        except HttpError as e:
            log.warning("Gmail messages.get %s failed: %s", mid, e)
            continue
        except Exception as e:
            log.warning("Gmail messages.get %s failed: %s", mid, e)
            continue

        normalized = _normalize_message(raw)
        if normalized:
            messages.append(normalized)

    return messages


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return (h.get("value") or "").strip()
    return ""


def _normalize_message(raw: dict) -> dict | None:
    payload = raw.get("payload") or {}
    headers = payload.get("headers") or []

    subject = _header(headers, "Subject") or "(no subject)"
    from_addr = _header(headers, "From")
    to_addr = _header(headers, "To")
    body = _extract_body_from_payload(payload).strip()

    if not body:
        snippet = raw.get("snippet") or ""
        body = snippet.strip()

    return {
        "id": raw.get("id", ""),
        "threadId": raw.get("threadId", ""),
        "internalDate": raw.get("internalDate", ""),
        "subject": subject,
        "from": from_addr,
        "to": to_addr,
        "body": body,
    }


def _decode_body_data(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_body_from_payload(payload: dict) -> str:
    mime = payload.get("mimeType", "")
    body_data = (payload.get("body") or {}).get("data")
    if body_data:
        text = _decode_body_data(body_data)
        if mime == "text/html":
            return _html_to_text(text)
        return text.strip()

    parts = payload.get("parts") or []
    plain_parts: list[str] = []
    html_parts: list[str] = []

    for part in parts:
        part_mime = part.get("mimeType", "")
        if part_mime.startswith("multipart/"):
            sub = _extract_body_from_payload(part)
            if sub:
                plain_parts.append(sub)
            continue
        sub = _extract_body_from_payload(part)
        if not sub:
            continue
        if part_mime == "text/plain":
            plain_parts.append(sub)
        elif part_mime == "text/html":
            html_parts.append(_html_to_text(sub))

    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return "\n".join(html_parts).strip()
    return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _format_date(message: dict) -> str:
    ms = message.get("internalDate")
    if ms:
        try:
            dt = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            pass
    return ""


def _chunk_text_with_overlap(text: str) -> list[str]:
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


def message_to_chunks(message: dict) -> list[tuple[int, str, str]]:
    """Convert a message into embeddable + snippet chunk pairs."""
    subject = (message.get("subject") or "(no subject)").strip()
    from_addr = (message.get("from") or "").strip()
    to_addr = (message.get("to") or "").strip()
    when = _format_date(message)
    body = (message.get("body") or "").strip()

    prefix_parts = [f"Mail: {subject}"]
    if from_addr:
        prefix_parts.append(f"From: {from_addr}")
    if to_addr:
        prefix_parts.append(f"To: {to_addr}")
    if when:
        prefix_parts.append(f"Date: {when}")
    prefix = f"[{' | '.join(prefix_parts)}]"

    snippet = subject
    if body:
        snippet = f"{subject}\n\n{body}"

    embed_body = subject
    if body:
        embed_body = f"{subject}\n\n{body}"

    if len(embed_body.split()) <= CHUNK_SIZE:
        embeddable = f"{prefix}\n{embed_body}" if prefix else embed_body
        return [(0, embeddable, snippet)]

    out: list[tuple[int, str, str]] = []
    for i, piece in enumerate(_chunk_text_with_overlap(embed_body)):
        embeddable = f"{prefix}\n{piece}" if prefix else piece
        out.append((i, embeddable, snippet if i == 0 else piece))
    return out


def _message_source_path(message: dict) -> str:
    """Stable id — one row per message (not per thread)."""
    return f"gmail://{message.get('id', '')}"


def _message_timestamp(message: dict) -> str:
    ms = message.get("internalDate")
    if ms:
        try:
            dt = datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
            return dt.isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return ""
