"""
Unit tests for core/checkpoint.py
"""

import json
import pytest
from pathlib import Path
from core.checkpoint import load, save, is_complete, delete, _fresh


class TestFresh:
    def test_default_keys(self):
        data = _fresh()
        assert data["offset"] == 0
        assert data["total"] is None
        assert data["successful_ids"] == []
        assert data["failed_ids"] == {}


class TestLoad:
    def test_returns_fresh_when_file_missing(self, tmp_path):
        result = load(tmp_path / "nonexistent.json")
        assert result == _fresh()

    def test_loads_existing_file(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        payload = {"offset": 50, "total": 200, "successful_ids": [1, 2], "failed_ids": {}}
        ckpt.write_text(json.dumps(payload))
        result = load(ckpt)
        assert result["offset"] == 50
        assert result["total"] == 200
        assert result["successful_ids"] == [1, 2]

    def test_accepts_str_path(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        ckpt.write_text(json.dumps({"offset": 10, "total": 100, "successful_ids": [], "failed_ids": {}}))
        result = load(str(ckpt))
        assert result["offset"] == 10

    def test_returns_fresh_on_corrupt_json(self, tmp_path):
        ckpt = tmp_path / "corrupt.json"
        ckpt.write_text("not valid json {{{")
        result = load(ckpt)
        assert result == _fresh()

    def test_missing_keys_filled_with_defaults(self, tmp_path):
        ckpt = tmp_path / "partial.json"
        ckpt.write_text(json.dumps({"offset": 5}))
        result = load(ckpt)
        assert result["offset"] == 5
        assert result["total"] is None
        assert result["successful_ids"] == []


class TestSave:
    def test_file_created(self, tmp_path):
        ckpt = tmp_path / "sub" / "ckpt.json"
        save(ckpt, {"offset": 0, "total": 10, "successful_ids": [], "failed_ids": {}})
        assert ckpt.exists()

    def test_round_trip(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        data = {"offset": 75, "total": 100, "successful_ids": [1, 2, 3], "failed_ids": {"4": 1}}
        save(ckpt, data)
        loaded = load(ckpt)
        assert loaded["offset"] == 75
        assert loaded["total"] == 100
        assert loaded["successful_ids"] == [1, 2, 3]

    def test_no_tmp_file_left_behind(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, _fresh())
        tmp = ckpt.with_suffix(".tmp")
        assert not tmp.exists()

    def test_timestamp_added(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, _fresh())
        raw = json.loads(ckpt.read_text())
        assert "_saved_at" in raw

    def test_accepts_str_path(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(str(ckpt), _fresh())
        assert ckpt.exists()


class TestIsComplete:
    def test_true_when_all_downloaded(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, {"offset": 3, "total": 3, "successful_ids": [1, 2, 3], "failed_ids": {}})
        assert is_complete(ckpt, total=3) is True

    def test_false_when_partial(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, {"offset": 1, "total": 3, "successful_ids": [1], "failed_ids": {}})
        assert is_complete(ckpt, total=3) is False

    def test_false_when_file_missing(self, tmp_path):
        assert is_complete(tmp_path / "missing.json", total=10) is False

    def test_true_when_more_than_total(self, tmp_path):
        """Edge case: allow > total (e.g. from a previous larger run)."""
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, {"offset": 5, "total": 5, "successful_ids": list(range(10)), "failed_ids": {}})
        assert is_complete(ckpt, total=5) is True


class TestDelete:
    def test_deletes_existing_file(self, tmp_path):
        ckpt = tmp_path / "ckpt.json"
        save(ckpt, _fresh())
        assert ckpt.exists()
        delete(ckpt)
        assert not ckpt.exists()

    def test_no_error_when_file_missing(self, tmp_path):
        delete(tmp_path / "ghost.json")  # should not raise


# ---------------------------------------------------------------------------
# Sync state
# ---------------------------------------------------------------------------

from core.checkpoint import (
    save_sync_state, load_sync_state, get_last_synced_at, sync_state_path
)


class TestSyncState:
    def test_load_returns_none_when_no_file(self, tmp_path):
        state = load_sync_state(tmp_path, "myproject")
        assert state["last_synced_at"] is None
        assert state["sync_count"] == 0

    def test_save_and_load_roundtrip(self, tmp_path):
        save_sync_state(tmp_path, "myproject", "2025-12-02T00:00:00Z")
        state = load_sync_state(tmp_path, "myproject")
        assert state["last_synced_at"] == "2025-12-02T00:00:00Z"

    def test_sync_count_increments(self, tmp_path):
        save_sync_state(tmp_path, "proj", "2025-01-01T00:00:00Z")
        save_sync_state(tmp_path, "proj", "2025-06-01T00:00:00Z")
        state = load_sync_state(tmp_path, "proj")
        assert state["sync_count"] == 2

    def test_latest_date_overwrites_previous(self, tmp_path):
        save_sync_state(tmp_path, "proj", "2025-01-01T00:00:00Z")
        save_sync_state(tmp_path, "proj", "2025-12-01T00:00:00Z")
        assert get_last_synced_at(tmp_path, "proj") == "2025-12-01T00:00:00Z"

    def test_get_last_synced_at_none_when_missing(self, tmp_path):
        assert get_last_synced_at(tmp_path, "ghost") is None

    def test_get_last_synced_at_returns_string(self, tmp_path):
        save_sync_state(tmp_path, "proj", "2026-01-15T12:00:00Z")
        result = get_last_synced_at(tmp_path, "proj")
        assert isinstance(result, str)
        assert result == "2026-01-15T12:00:00Z"

    def test_defaults_to_now_when_no_date_given(self, tmp_path):
        save_sync_state(tmp_path, "proj")
        result = get_last_synced_at(tmp_path, "proj")
        assert result is not None
        assert "T" in result  # ISO-8601 format

    def test_different_projects_isolated(self, tmp_path):
        save_sync_state(tmp_path, "proj_a", "2025-01-01T00:00:00Z")
        save_sync_state(tmp_path, "proj_b", "2025-06-01T00:00:00Z")
        assert get_last_synced_at(tmp_path, "proj_a") == "2025-01-01T00:00:00Z"
        assert get_last_synced_at(tmp_path, "proj_b") == "2025-06-01T00:00:00Z"

    def test_state_file_path(self, tmp_path):
        path = sync_state_path(tmp_path, "qesecurity")
        assert path.name == "qesecurity_sync.json"
        assert path.parent == tmp_path

    def test_corrupt_file_returns_defaults(self, tmp_path):
        path = sync_state_path(tmp_path, "proj")
        path.write_text("not valid json")
        state = load_sync_state(tmp_path, "proj")
        assert state["last_synced_at"] is None
