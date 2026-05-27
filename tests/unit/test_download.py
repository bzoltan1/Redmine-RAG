"""
Unit tests for the merge logic used by pipeline/01_download.py.

The download/sync pipeline scripts are thin wrappers; the only pure
logic worth unit-testing independently is the merge_into helper.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# merge_into lives in the pipeline script; import it directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dl", Path(__file__).parent.parent.parent / "pipeline" / "01_download.py"
)
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)  # type: ignore
merge_into = dl.merge_into


class TestMergeInto:
    def _issue(self, iid: int, subject: str = "s", updated: str = "2025-01-01") -> dict:
        return {"id": iid, "subject": subject, "updated_on": updated}

    def test_new_issue_added(self):
        existing = {1: self._issue(1)}
        n_new, n_upd = merge_into(existing, [self._issue(2)])
        assert 2 in existing
        assert n_new == 1
        assert n_upd == 0

    def test_existing_issue_replaced(self):
        existing = {1: self._issue(1, subject="old")}
        n_new, n_upd = merge_into(existing, [self._issue(1, subject="new")])
        assert existing[1]["subject"] == "new"
        assert n_new == 0
        assert n_upd == 1

    def test_mix_of_new_and_updated(self):
        existing = {1: self._issue(1), 2: self._issue(2)}
        incoming = [self._issue(2, subject="updated"), self._issue(3)]
        n_new, n_upd = merge_into(existing, incoming)
        assert n_new == 1
        assert n_upd == 1
        assert len(existing) == 3

    def test_empty_incoming_no_change(self):
        existing = {1: self._issue(1)}
        n_new, n_upd = merge_into(existing, [])
        assert n_new == 0
        assert n_upd == 0
        assert len(existing) == 1

    def test_empty_existing_all_new(self):
        existing = {}
        incoming = [self._issue(i) for i in range(1, 6)]
        n_new, n_upd = merge_into(existing, incoming)
        assert n_new == 5
        assert n_upd == 0
        assert len(existing) == 5

    def test_original_unaffected_issues_preserved(self):
        existing = {1: self._issue(1), 2: self._issue(2), 3: self._issue(3)}
        merge_into(existing, [self._issue(2, subject="updated")])
        assert existing[1]["subject"] == "s"
        assert existing[3]["subject"] == "s"
