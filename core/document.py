"""
core/document.py — Redmine issue -> ChromaDB document conversion.

Supports two modes:

  prepare(issue)
    Single-document mode (one doc per issue). Used when the issue is short
    enough that truncation is not a concern. Kept for backward compatibility
    and simple cases.

  prepare_chunks(issue, journals_per_chunk, max_text_len)
    Chunking mode. Splits a long issue into multiple documents:
      - Chunk 0: subject + description + metadata summary header
      - Chunk N: every `journals_per_chunk` journal entries

    All chunks share the same parent issue_id in their metadata so that
    deduplication by parent can be applied at retrieval time.

    Chunk IDs use the format: "issue_<id>_chunk_<n>"
    Chunk 0 always uses: "issue_<id>" (for backward compatibility with
    existing queries that reference issue IDs directly).

No file I/O.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metadata_base(issue: dict[str, Any], journals: list[dict]) -> dict[str, Any]:
    """Build the metadata dict shared by all chunks of an issue."""
    assigned = issue.get("assigned_to") or {}
    return {
        "issue_id":    str(issue.get("id") or ""),
        "subject":     (issue.get("subject") or "")[:500],
        "status":      (issue.get("status") or {}).get("name") or "",
        "priority":    (issue.get("priority") or {}).get("name") or "",
        "tracker":     (issue.get("tracker") or {}).get("name") or "",
        "project":     (issue.get("project") or {}).get("name") or "",
        "project_id":  issue.get("project_identifier") or "",
        "created_on":  issue.get("created_on") or "",
        "updated_on":  issue.get("updated_on") or "",
        "author":      (issue.get("author") or {}).get("name") or "",
        "assigned_to": assigned.get("name") or "",
        "num_journals": len(journals),
    }


def _format_journals(journal_slice: list[dict]) -> str:
    """Render a list of journal entries as a text block."""
    lines: list[str] = []
    for j in journal_slice:
        notes = j.get("notes")
        if not notes:
            continue
        user_name = (j.get("user") or {}).get("name") or "Unknown"
        lines.append(f"- {user_name}: {notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def prepare(issue: dict[str, Any], max_text_len: int = 8192) -> dict[str, Any]:
    """
    Convert a Redmine issue dict to a single document ready for ChromaDB.

    Truncates the combined text at *max_text_len* characters. For issues with
    many journal entries, use prepare_chunks() instead to avoid losing history.

    Returns
    -------
    A dict with keys: id (str), text (str), metadata (dict).
    """
    subject: str = issue.get("subject") or ""
    description: str = issue.get("description") or ""

    parts: list[str] = [f"Subject: {subject}"]
    if description:
        parts.append(f"Description: {description}")

    journals: list[dict] = issue.get("journals") or []
    comment_lines = _format_journals(journals)
    if comment_lines:
        parts.append("Comments:")
        parts.append(comment_lines)

    text = "\n".join(parts).strip()
    if len(text) > max_text_len:
        text = text[:max_text_len] + "\n...[truncated]"

    return {
        "id":       f"issue_{issue.get('id') or ''}",
        "text":     text,
        "metadata": _metadata_base(issue, journals),
    }


def prepare_chunks(
    issue: dict[str, Any],
    journals_per_chunk: int = 5,
    max_text_len: int = 4000,
) -> list[dict[str, Any]]:
    """
    Split a Redmine issue into multiple overlapping chunks for embedding.

    Chunk layout
    ------------
    Chunk 0 — "issue_<id>":
        Subject + Description.  Always produced, even for empty issues.
        Metadata includes chunk_index=0, is_description=True.

    Chunk N (N >= 1) — "issue_<id>_chunk_<N>":
        Subject (repeated for context) + up to `journals_per_chunk` journal
        entries.  Only produced when journals exist.
        Metadata includes chunk_index=N, is_description=False.

    Each chunk is independently truncated to *max_text_len* characters so
    it fits within the embedding model's context window.

    Parameters
    ----------
    issue:              Redmine issue dict.
    journals_per_chunk: Journal entries per chunk (default 5).
    max_text_len:       Character cap per chunk (default 4000).

    Returns
    -------
    List of chunk dicts (id, text, metadata). Always at least one element.
    """
    subject: str = issue.get("subject") or ""
    description: str = issue.get("description") or ""
    issue_id = issue.get("id") or ""
    journals: list[dict] = issue.get("journals") or []
    base_meta = _metadata_base(issue, journals)
    chunks: list[dict[str, Any]] = []

    # --- Chunk 0: description ---
    desc_parts = [f"Subject: {subject}"]
    if description:
        desc_parts.append(f"Description: {description}")
    desc_text = "\n".join(desc_parts).strip()
    if len(desc_text) > max_text_len:
        desc_text = desc_text[:max_text_len] + "\n...[truncated]"

    chunks.append({
        "id": f"issue_{issue_id}",
        "text": desc_text,
        "metadata": {**base_meta, "chunk_index": 0, "is_description": True},
    })

    # --- Journal chunks ---
    # Only include journals that have actual notes
    noted = [j for j in journals if j.get("notes")]
    for chunk_idx, start in enumerate(range(0, len(noted), journals_per_chunk), start=1):
        slice_ = noted[start : start + journals_per_chunk]
        journal_text = _format_journals(slice_)
        if not journal_text:
            continue
        chunk_text = f"Subject: {subject}\nComments:\n{journal_text}"
        if len(chunk_text) > max_text_len:
            chunk_text = chunk_text[:max_text_len] + "\n...[truncated]"
        chunks.append({
            "id": f"issue_{issue_id}_chunk_{chunk_idx}",
            "text": chunk_text,
            "metadata": {**base_meta, "chunk_index": chunk_idx, "is_description": False},
        })

    return chunks
