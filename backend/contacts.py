# -*- coding: utf-8 -*-
"""
Resolve iMessage handle identifiers (phone numbers / emails) to display names
using macOS's local Contacts (AddressBook) database.

chat.db's `handle.id` is always a raw phone number or email — it has no idea
about contact names. Those live in
~/Library/Application Support/AddressBook/AddressBook-v22.abcddb and, for
linked accounts (iCloud, Google, etc.), in
~/Library/Application Support/AddressBook/Sources/<uuid>/AddressBook-v22.abcddb

Reading these requires the same Full Disk Access grant as chat.db.
"""
import re
import sqlite3
from pathlib import Path

_ADDRESSBOOK_ROOT = Path.home() / "Library" / "Application Support" / "AddressBook"


def _addressbook_dbs() -> list[Path]:
    dbs = []
    main_db = _ADDRESSBOOK_ROOT / "AddressBook-v22.abcddb"
    if main_db.exists():
        dbs.append(main_db)
    sources_dir = _ADDRESSBOOK_ROOT / "Sources"
    if sources_dir.is_dir():
        for source_db in sources_dir.glob("*/AddressBook-v22.abcddb"):
            if source_db.exists():
                dbs.append(source_db)
    return dbs


def _normalize_phone(raw: str) -> str:
    """Strip everything but digits and keep the last 10 (US-style local
    number). chat.db handles are usually E.164 (+15551234567); AddressBook
    numbers are often formatted ("(555) 123-4567"). Comparing the last 10
    digits matches both regardless of country-code/formatting differences."""
    digits = re.sub(r"\D", "", raw)
    return digits[-10:] if len(digits) >= 10 else digits


def _full_name(first: str | None, last: str | None) -> str:
    return " ".join(p for p in (first, last) if p)


def load_contact_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (phone_map, email_map):
      phone_map: normalized last-10-digit number -> display name
      email_map: lowercased email address -> display name
    Empty dicts if no AddressBook database is readable."""
    phone_map: dict[str, str] = {}
    email_map: dict[str, str] = {}

    for db_path in _addressbook_dbs():
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cur = con.cursor()

            for first, last, number in cur.execute("""
                SELECT r.ZFIRSTNAME, r.ZLASTNAME, p.ZFULLNUMBER
                FROM ZABCDPHONENUMBER p
                JOIN ZABCDRECORD r ON r.Z_PK = p.ZOWNER
                WHERE p.ZFULLNUMBER IS NOT NULL
            """):
                name = _full_name(first, last)
                if not name:
                    continue
                key = _normalize_phone(number)
                if key:
                    phone_map.setdefault(key, name)

            for first, last, address in cur.execute("""
                SELECT r.ZFIRSTNAME, r.ZLASTNAME, e.ZADDRESS
                FROM ZABCDEMAILADDRESS e
                JOIN ZABCDRECORD r ON r.Z_PK = e.ZOWNER
                WHERE e.ZADDRESS IS NOT NULL
            """):
                name = _full_name(first, last)
                if not name:
                    continue
                key = address.strip().lower()
                if key:
                    email_map.setdefault(key, name)

            con.close()
        except sqlite3.Error:
            continue

    return phone_map, email_map


def resolve_contact_name(
    handle: str,
    phone_map: dict[str, str],
    email_map: dict[str, str],
) -> str | None:
    """Look up a chat.db handle.id (phone or email) in the contact maps.
    Returns None if no match is found."""
    if "@" in handle:
        return email_map.get(handle.strip().lower())
    return phone_map.get(_normalize_phone(handle))
