"""
core/checkpoint.py — Checkpoint and sync-state persistence.

Two distinct concerns:

Download checkpoints
    Track which issues have been fetched in the current run so that an
    interrupted download can resume from where it left off.
    File: data/raw/<project_id>_checkpoint.json

Sync state
    Record the timestamp of the last successful full or incremental sync
    for each project. Used by --sync mode to automatically determine the
    --since date without requiring the user to remember it.
    File: data/raw/<project_id>_sync.json

No business logic beyond load/save/query — no HTTP, no ChromaDB.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEFAULT: dict[str, Any] = {
    "offset": 0,
    "total": None,
    "successful_ids": [],
    "failed_ids": {},   # {issue_id_str: retry_count}
}


def load(path: Path | str) -> dict[str, Any]:
    """
    Load a checkpoint file and return its contents.

    If the file does not exist, returns a fresh default checkpoint dict.
    """
    path = Path(path)
    if not path.exists():
        return _fresh()

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return _fresh()

    # Normalise: ensure all expected keys are present
    result = _fresh()
    result.update(data)
    # failed_ids keys come back from JSON as strings; keep as strings
    return result


def save(path: Path | str, data: dict[str, Any]) -> None:
    """
    Atomically write checkpoint data to *path*.

    Writes to a temporary sibling file then renames to avoid a corrupt
    checkpoint if the process is killed mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(".tmp")
    payload = dict(data)
    payload["_saved_at"] = time.time()

    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    tmp.replace(path)


def is_complete(path: Path | str, total: int) -> bool:
    """
    Return True if the checkpoint at *path* records that all *total* issues
    have been successfully downloaded.
    """
    data = load(path)
    successful = data.get("successful_ids") or []
    return len(successful) >= total


def delete(path: Path | str) -> None:
    """Remove the checkpoint file if it exists."""
    path = Path(path)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------

def sync_state_path(raw_dir: Path, project_id: str) -> Path:
    """Return the sync state file path for *project_id*."""
    return raw_dir / f"{project_id}_sync.json"


def load_sync_state(raw_dir: Path, project_id: str) -> dict[str, Any]:
    """
    Load the sync state for *project_id*.

    Returns a dict with at least:
        last_synced_at  — ISO-8601 UTC string, or None if never synced
        sync_count      — number of syncs performed
    """
    path = sync_state_path(raw_dir, project_id)
    if not path.exists():
        return {"last_synced_at": None, "sync_count": 0}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"last_synced_at": None, "sync_count": 0}


def save_sync_state(raw_dir: Path, project_id: str, synced_at: str | None = None) -> None:
    """
    Record a successful sync for *project_id*.

    Parameters
    ----------
    raw_dir:    Directory where sync state files live.
    project_id: Redmine project identifier.
    synced_at:  ISO-8601 UTC timestamp string to record. Defaults to now.
    """
    path = sync_state_path(raw_dir, project_id)
    state = load_sync_state(raw_dir, project_id)

    if synced_at is None:
        synced_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    state["last_synced_at"] = synced_at
    state["sync_count"] = state.get("sync_count", 0) + 1
    state["_saved_at"] = time.time()

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def get_last_synced_at(raw_dir: Path, project_id: str) -> str | None:
    """
    Return the ISO-8601 timestamp of the last successful sync, or None.
    """
    return load_sync_state(raw_dir, project_id).get("last_synced_at")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fresh() -> dict[str, Any]:
    """Return a fresh, empty checkpoint dict."""
    return {
        "offset": 0,
        "total": None,
        "successful_ids": [],
        "failed_ids": {},
    }
