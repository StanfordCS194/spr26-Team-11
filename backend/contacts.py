"""Resolve iMessage handle IDs (phone/email) to macOS Contacts display names."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ADDRESSBOOK_ROOT = Path.home() / "Library" / "Application Support" / "AddressBook"


def _normalize_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def _normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def _record_name(first: str | None, last: str | None, org: str | None) -> str:
    parts = [p for p in (first, last) if p and p.strip()]
    if parts:
        return " ".join(parts)
    return (org or "").strip()


def _find_richest_addressbook() -> Path | None:
    candidates = list(ADDRESSBOOK_ROOT.glob("**/AddressBook-v22.abcddb"))
    legacy = ADDRESSBOOK_ROOT / "AddressBook-v22.abcddb"
    if legacy.exists() and legacy not in candidates:
        candidates.append(legacy)

    best: Path | None = None
    best_count = 0
    for path in candidates:
        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            count = con.execute("SELECT COUNT(*) FROM ZABCDRECORD").fetchone()[0]
            con.close()
        except sqlite3.Error:
            continue
        if count > best_count:
            best_count = count
            best = path
    return best


def load_handle_display_names() -> dict[str, str]:
    """Map normalized phone digits (last 10) or lowercase email -> display name."""
    db_path = _find_richest_addressbook()
    if not db_path:
        return {}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    lookup: dict[str, str] = {}

    cur.execute("""
        SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION, p.ZFULLNUMBER
        FROM ZABCDRECORD r
        JOIN ZABCDPHONENUMBER p ON p.ZOWNER = r.Z_PK
        WHERE p.ZFULLNUMBER IS NOT NULL
    """)
    for first, last, org, number in cur.fetchall():
        name = _record_name(first, last, org)
        if not name:
            continue
        key = _normalize_phone(number)
        if key:
            lookup.setdefault(key, name)

    cur.execute("""
        SELECT r.ZFIRSTNAME, r.ZLASTNAME, r.ZORGANIZATION, e.ZADDRESS
        FROM ZABCDRECORD r
        JOIN ZABCDEMAILADDRESS e ON e.ZOWNER = r.Z_PK
        WHERE e.ZADDRESS IS NOT NULL
    """)
    for first, last, org, email in cur.fetchall():
        name = _record_name(first, last, org)
        if not name:
            continue
        key = _normalize_email(email)
        if key:
            lookup.setdefault(key, name)

    con.close()
    return lookup


def lookup_display_name(handle_id: str, lookup: dict[str, str]) -> str:
    if "@" in handle_id:
        return lookup.get(_normalize_email(handle_id), "")
    return lookup.get(_normalize_phone(handle_id), "")


def format_contact_label(handle_id: str, lookup: dict[str, str]) -> str:
    name = lookup_display_name(handle_id, lookup)
    if name:
        return f"{name} ({handle_id})"
    return handle_id
