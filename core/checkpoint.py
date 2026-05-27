"""
core/checkpoint.py — Checkpoint persistence for the download pipeline stage.

Checkpoints track which issues have been successfully downloaded so that
a failed or interrupted run can resume from where it left off.
No business logic beyond load/save/query — no HTTP, no ChromaDB.
"""

import json
import time
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
