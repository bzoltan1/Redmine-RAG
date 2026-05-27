"""
Unit tests for core/document.py — covers both prepare() and prepare_chunks().
"""

import pytest
from tests.conftest import make_issue, make_journal
from core.document import prepare, prepare_chunks


# ---------------------------------------------------------------------------
# prepare() — single-document mode (existing tests, kept intact)
# ---------------------------------------------------------------------------

class TestPrepareId:
    def test_id_format(self, simple_issue):
        doc = prepare(simple_issue)
        assert doc["id"] == f"issue_{simple_issue['id']}"

    def test_id_when_missing(self):
        doc = prepare({})
        assert doc["id"] == "issue_"


class TestPrepareText:
    def test_subject_always_present(self, simple_issue):
        doc = prepare(simple_issue)
        assert "Subject: Test issue subject" in doc["text"]

    def test_description_included(self, simple_issue):
        doc = prepare(simple_issue)
        assert "Description: Test description" in doc["text"]

    def test_empty_description_omitted(self):
        issue = make_issue(description="")
        doc = prepare(issue)
        assert "Description:" not in doc["text"]

    def test_none_description_omitted(self):
        issue = make_issue(description=None)
        doc = prepare(issue)
        assert "Description:" not in doc["text"]

    def test_comments_included(self, issue_with_journals):
        doc = prepare(issue_with_journals)
        assert "Comments:" in doc["text"]
        assert "First comment." in doc["text"]
        assert "Second comment." in doc["text"]

    def test_empty_notes_skipped(self):
        journals = [
            make_journal(notes="Real note"),
            make_journal(journal_id=2, notes=""),
            make_journal(journal_id=3, notes=None),
        ]
        issue = make_issue(journals=journals)
        doc = prepare(issue)
        assert "Real note" in doc["text"]
        assert doc["text"].count("- Bob:") == 1

    def test_no_journals_no_comments_header(self, simple_issue):
        doc = prepare(simple_issue)
        assert "Comments:" not in doc["text"]

    def test_journal_user_name_in_text(self, issue_with_journals):
        doc = prepare(issue_with_journals)
        assert "Bob" in doc["text"]
        assert "Carol" in doc["text"]

    def test_missing_journal_user_defaults_to_unknown(self):
        journals = [{"id": 1, "notes": "A note with no user field"}]
        issue = make_issue(journals=journals)
        doc = prepare(issue)
        assert "Unknown" in doc["text"]

    def test_truncation_applied(self):
        long_desc = "x" * 10_000
        issue = make_issue(description=long_desc)
        doc = prepare(issue, max_text_len=100)
        assert len(doc["text"]) <= 100 + len("\n...[truncated]")
        assert doc["text"].endswith("...[truncated]")

    def test_no_truncation_when_within_limit(self, simple_issue):
        doc = prepare(simple_issue, max_text_len=8192)
        assert "...[truncated]" not in doc["text"]

    def test_empty_issue_produces_subject_line(self):
        doc = prepare({})
        assert "Subject:" in doc["text"]


class TestPrepareMetadata:
    def test_all_keys_present(self, simple_issue):
        meta = prepare(simple_issue)["metadata"]
        expected_keys = {
            "issue_id", "subject", "status", "priority", "tracker",
            "project", "project_id", "created_on", "updated_on",
            "author", "assigned_to", "num_journals",
        }
        assert expected_keys == set(meta.keys())

    def test_issue_id_is_string(self, simple_issue):
        meta = prepare(simple_issue)["metadata"]
        assert isinstance(meta["issue_id"], str)
        assert meta["issue_id"] == str(simple_issue["id"])

    def test_num_journals_correct(self, issue_with_journals):
        meta = prepare(issue_with_journals)["metadata"]
        assert meta["num_journals"] == 2

    def test_num_journals_zero_when_none(self, simple_issue):
        meta = prepare(simple_issue)["metadata"]
        assert meta["num_journals"] == 0

    def test_assigned_to_empty_string_when_none(self, simple_issue):
        meta = prepare(simple_issue)["metadata"]
        assert meta["assigned_to"] == ""

    def test_assigned_to_name_when_present(self, issue_full):
        meta = prepare(issue_full)["metadata"]
        assert meta["assigned_to"] == "Bob"

    def test_subject_truncated_to_500_in_metadata(self):
        long_subject = "A" * 600
        issue = make_issue(subject=long_subject)
        meta = prepare(issue)["metadata"]
        assert len(meta["subject"]) == 500

    def test_missing_nested_fields_default_to_empty(self):
        issue = {"id": 99}
        meta = prepare(issue)["metadata"]
        assert meta["status"] == ""
        assert meta["priority"] == ""
        assert meta["tracker"] == ""
        assert meta["project"] == ""
        assert meta["author"] == ""
        assert meta["assigned_to"] == ""

    def test_project_id_from_project_identifier(self, simple_issue):
        meta = prepare(simple_issue)["metadata"]
        assert meta["project_id"] == simple_issue["project_identifier"]


# ---------------------------------------------------------------------------
# prepare_chunks() — section-based chunking mode
# ---------------------------------------------------------------------------

def make_issue_with_n_journals(n: int, subject: str = "Test subject") -> dict:
    journals = [
        make_journal(journal_id=i, notes=f"Journal entry number {i}.")
        for i in range(1, n + 1)
    ]
    return make_issue(subject=subject, journals=journals)


class TestPrepareChunksBasic:
    def test_returns_list(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert isinstance(chunks, list)

    def test_always_at_least_one_chunk(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert len(chunks) >= 1

    def test_empty_issue_produces_one_chunk(self):
        chunks = prepare_chunks({})
        assert len(chunks) == 1

    def test_chunk_zero_id_equals_issue_id(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert chunks[0]["id"] == f"issue_{simple_issue['id']}"

    def test_subsequent_chunks_have_numbered_ids(self):
        # 6 journals, 5 per chunk → description chunk + chunk_1(5) + chunk_2(1) = 3 total
        issue = make_issue_with_n_journals(6)
        chunks = prepare_chunks(issue, journals_per_chunk=5)
        assert len(chunks) == 3
        assert chunks[1]["id"] == f"issue_{issue['id']}_chunk_1"
        assert chunks[2]["id"] == f"issue_{issue['id']}_chunk_2"

    def test_no_journals_produces_one_chunk(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert len(chunks) == 1


class TestPrepareChunksContent:
    def test_chunk_zero_contains_description(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert "Description:" in chunks[0]["text"]

    def test_chunk_zero_contains_subject(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert simple_issue["subject"] in chunks[0]["text"]

    def test_journal_chunks_contain_subject_for_context(self):
        issue = make_issue_with_n_journals(6, subject="My subject")
        chunks = prepare_chunks(issue, journals_per_chunk=5)
        # All journal chunks repeat subject for embedding context
        assert "My subject" in chunks[1]["text"]

    def test_journal_entries_distributed_across_chunks(self):
        issue = make_issue_with_n_journals(10)
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        # chunk 0: description; chunks 1-4: journals (ceil(10/3)=4)
        assert len(chunks) == 5

    def test_all_journal_notes_appear_somewhere(self):
        issue = make_issue_with_n_journals(7)
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        all_text = " ".join(c["text"] for c in chunks)
        for i in range(1, 8):
            assert f"Journal entry number {i}." in all_text

    def test_empty_notes_not_included(self):
        journals = [
            make_journal(journal_id=1, notes="Real note"),
            make_journal(journal_id=2, notes=""),
            make_journal(journal_id=3, notes=None),
        ]
        issue = make_issue(journals=journals)
        chunks = prepare_chunks(issue, journals_per_chunk=5)
        all_text = " ".join(c["text"] for c in chunks)
        assert "Real note" in all_text
        assert all_text.count("- Bob:") == 1

    def test_chunk_text_truncated_to_max_len(self):
        long_desc = "y" * 10_000
        issue = make_issue(description=long_desc)
        chunks = prepare_chunks(issue, max_text_len=200)
        assert chunks[0]["text"].endswith("...[truncated]")
        assert len(chunks[0]["text"]) <= 200 + len("\n...[truncated]")


class TestPrepareChunksMetadata:
    def test_all_chunks_share_issue_id(self):
        issue = make_issue_with_n_journals(6)
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        for chunk in chunks:
            assert chunk["metadata"]["issue_id"] == str(issue["id"])

    def test_chunk_zero_is_description_flag(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert chunks[0]["metadata"]["is_description"] is True

    def test_journal_chunks_not_description_flag(self):
        issue = make_issue_with_n_journals(6)
        chunks = prepare_chunks(issue, journals_per_chunk=5)
        assert chunks[1]["metadata"]["is_description"] is False

    def test_chunk_index_increments(self):
        issue = make_issue_with_n_journals(10)
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        for i, chunk in enumerate(chunks):
            assert chunk["metadata"]["chunk_index"] == i

    def test_num_journals_consistent_across_chunks(self):
        issue = make_issue_with_n_journals(8)
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        num_j = issue["id"]  # just to verify it's same in all chunks
        for chunk in chunks:
            assert chunk["metadata"]["num_journals"] == 8

    def test_status_present_in_all_chunks(self):
        issue = make_issue_with_n_journals(6, subject="Sub")
        chunks = prepare_chunks(issue, journals_per_chunk=3)
        for chunk in chunks:
            assert "status" in chunk["metadata"]

    def test_chunk_metadata_has_chunk_index_key(self, simple_issue):
        chunks = prepare_chunks(simple_issue)
        assert "chunk_index" in chunks[0]["metadata"]
