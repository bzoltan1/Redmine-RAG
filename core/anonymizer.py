"""
core/anonymizer.py — User field anonymization for Redmine issue dicts.

Two-pass anonymization:
  1. Structured fields — replace user objects (author, assigned_to,
     journal users, watchers) with User_XXXXX placeholders.
  2. Free-text PII scan — redact email addresses, IPv4 addresses, and
     likely internal hostnames from description and journal notes fields.
     Also strips openQA test-run URLs (openqa.suse.de / openqa.opensuse.org)
     which add noise without semantic value.
     Matches are logged so they can be reviewed; content is not dropped.

Pure data transformation — no file I/O.
"""

import copy
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# Email addresses: user@host.tld
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# IPv4 addresses: 1.2.3.4 (not matched inside version strings like 1.2.3.4-5)
_IPV4_RE = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
)

# Internal hostnames: word chars, hyphens, dots — at least two labels,
# ending in a known internal TLD or common patterns like .suse.de, .lab, etc.
# Intentionally conservative: only match if it looks like a real hostname.
_HOSTNAME_RE = re.compile(
    r"\b(?:[a-zA-Z0-9\-]+\.){1,}(?:suse\.de|suse\.com|lab|internal|local|intranet)\b",
    re.IGNORECASE,
)

# openQA test-run URLs — these are high-volume, machine-generated links that
# add noise without semantic value.  Strip the full URL so the surrounding
# prose remains readable.
# Matches both http and https, with or without a trailing path/query.
# Examples:
#   https://openqa.suse.de/tests/12345678
#   http://openqa.opensuse.org/tests/12345#step/boot/1
_OPENQA_URL_RE = re.compile(
    r"https?://openqa\.(?:suse\.de|opensuse\.org)\S*",
    re.IGNORECASE,
)

_REDACT_EMAIL    = "[REDACTED-EMAIL]"
_REDACT_IPV4     = "[REDACTED-IP]"
_REDACT_HOSTNAME = "[REDACTED-HOST]"
_REDACT_OPENQA   = "[OPENQA-URL]"


def generate_anonymous_name(user_id: int) -> str:
    """Return a deterministic anonymous name for a given user ID."""
    return f"User_{user_id:05d}"


def anonymize_user(
    user_obj: dict[str, Any],
    mapping: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """
    Replace the 'name' field in a user dict with an anonymous name.

    The mapping dict is updated in place so that subsequent calls with
    the same user_id return the same anonymous name.

    Parameters
    ----------
    user_obj: A dict with at least an 'id' key and optionally 'name'.
    mapping:  Accumulated {user_id: {original_name, anonymous_name}} dict.

    Returns
    -------
    A new dict with the same 'id' but an anonymized 'name'.
    """
    if not user_obj or "id" not in user_obj:
        return user_obj

    user_id: int = user_obj["id"]
    original_name: str = user_obj.get("name") or "Unknown"

    if user_id not in mapping:
        mapping[user_id] = {
            "original_name": original_name,
            "anonymous_name": generate_anonymous_name(user_id),
        }

    return {"id": user_id, "name": mapping[user_id]["anonymous_name"]}


def scrub_pii(text: str, issue_id: int | str = "?") -> str:
    """
    Redact emails, IPv4 addresses, and internal hostnames from *text*.

    Matches are replaced with bracketed placeholders. Each unique match
    is logged at DEBUG level so it can be reviewed if needed.

    Parameters
    ----------
    text:     Free-text string (description or journal note).
    issue_id: Parent issue ID, used only for log messages.

    Returns
    -------
    The text with PII patterns replaced.
    """
    if not text:
        return text

    def _replace(pattern: re.Pattern, replacement: str, src: str) -> str:
        matches = pattern.findall(src)
        for m in matches:
            logger.debug("PII redacted in issue #%s: %s -> %s", issue_id, m, replacement)
        return pattern.sub(replacement, src)

    text = _replace(_OPENQA_URL_RE, _REDACT_OPENQA,   text)
    text = _replace(_EMAIL_RE,      _REDACT_EMAIL,    text)
    text = _replace(_IPV4_RE,       _REDACT_IPV4,     text)
    text = _replace(_HOSTNAME_RE,   _REDACT_HOSTNAME, text)
    return text


def anonymize_issue(
    issue: dict[str, Any],
    mapping: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """
    Anonymize all user-identifying fields in a single issue dict.

    Fields touched: author, assigned_to, journals[*].user, watchers[*].

    Parameters
    ----------
    issue:   A Redmine issue dict (will not be mutated).
    mapping: Accumulated user mapping dict (mutated in place).

    Returns
    -------
    A deep-copied issue dict with user fields anonymized.
    """
    issue = copy.deepcopy(issue)
    issue_id = issue.get("id", "?")

    # Pass 1 — structured user fields
    for field in ("author", "assigned_to"):
        if issue.get(field):
            issue[field] = anonymize_user(issue[field], mapping)

    journals = issue.get("journals") or []
    for journal in journals:
        if journal.get("user"):
            journal["user"] = anonymize_user(journal["user"], mapping)

    watchers = issue.get("watchers") or []
    for i, watcher in enumerate(watchers):
        if isinstance(watcher, dict) and "id" in watcher:
            watchers[i] = anonymize_user(watcher, mapping)

    # Pass 2 — free-text PII scan on description and journal notes
    if issue.get("description"):
        issue["description"] = scrub_pii(issue["description"], issue_id)

    for journal in journals:
        if journal.get("notes"):
            journal["notes"] = scrub_pii(journal["notes"], issue_id)

    return issue
