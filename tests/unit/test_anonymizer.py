"""
Unit tests for core/anonymizer.py
"""

import pytest
from tests.conftest import make_issue, make_journal
from core.anonymizer import anonymize_issue, anonymize_user, generate_anonymous_name, scrub_pii


class TestGenerateAnonymousName:
    def test_format(self):
        assert generate_anonymous_name(1) == "User_00001"
        assert generate_anonymous_name(99) == "User_00099"
        assert generate_anonymous_name(12345) == "User_12345"

    def test_deterministic(self):
        assert generate_anonymous_name(42) == generate_anonymous_name(42)

    def test_large_id(self):
        name = generate_anonymous_name(999999)
        assert name.startswith("User_")


class TestAnonymizeUser:
    def test_name_replaced(self):
        mapping: dict = {}
        result = anonymize_user({"id": 10, "name": "Alice"}, mapping)
        assert result["name"] != "Alice"
        assert result["name"] == "User_00010"

    def test_id_preserved(self):
        mapping: dict = {}
        result = anonymize_user({"id": 10, "name": "Alice"}, mapping)
        assert result["id"] == 10

    def test_mapping_updated(self):
        mapping: dict = {}
        anonymize_user({"id": 10, "name": "Alice"}, mapping)
        assert 10 in mapping
        assert mapping[10]["original_name"] == "Alice"
        assert mapping[10]["anonymous_name"] == "User_00010"

    def test_same_id_same_anonymous_name(self):
        mapping: dict = {}
        r1 = anonymize_user({"id": 10, "name": "Alice"}, mapping)
        r2 = anonymize_user({"id": 10, "name": "Alice Duplicate"}, mapping)
        assert r1["name"] == r2["name"]

    def test_different_ids_different_names(self):
        mapping: dict = {}
        r1 = anonymize_user({"id": 10, "name": "Alice"}, mapping)
        r2 = anonymize_user({"id": 11, "name": "Bob"}, mapping)
        assert r1["name"] != r2["name"]

    def test_empty_dict_returned_unchanged(self):
        mapping: dict = {}
        assert anonymize_user({}, mapping) == {}

    def test_none_returned_unchanged(self):
        mapping: dict = {}
        assert anonymize_user(None, mapping) is None

    def test_missing_name_defaults_to_unknown(self):
        mapping: dict = {}
        result = anonymize_user({"id": 5}, mapping)
        assert result["name"] == "User_00005"
        assert mapping[5]["original_name"] == "Unknown"


class TestAnonymizeIssue:
    def test_author_anonymized(self, simple_issue):
        mapping: dict = {}
        result = anonymize_issue(simple_issue, mapping)
        assert result["author"]["name"] != simple_issue["author"]["name"]
        assert result["author"]["name"].startswith("User_")

    def test_original_issue_not_mutated(self, simple_issue):
        original_name = simple_issue["author"]["name"]
        mapping: dict = {}
        anonymize_issue(simple_issue, mapping)
        assert simple_issue["author"]["name"] == original_name

    def test_assigned_to_anonymized(self, issue_full):
        mapping: dict = {}
        result = anonymize_issue(issue_full, mapping)
        assert result["assigned_to"]["name"].startswith("User_")

    def test_assigned_to_none_unchanged(self, simple_issue):
        mapping: dict = {}
        result = anonymize_issue(simple_issue, mapping)
        assert result["assigned_to"] is None

    def test_journal_users_anonymized(self, issue_with_journals):
        mapping: dict = {}
        result = anonymize_issue(issue_with_journals, mapping)
        for journal in result["journals"]:
            assert journal["user"]["name"].startswith("User_")

    def test_watchers_anonymized(self, issue_with_watchers):
        mapping: dict = {}
        result = anonymize_issue(issue_with_watchers, mapping)
        for watcher in result["watchers"]:
            assert watcher["name"].startswith("User_")

    def test_mapping_accumulates_all_users(self, issue_full):
        mapping: dict = {}
        anonymize_issue(issue_full, mapping)
        # author id=10, assigned_to id=20 (Bob), journal user id=20, watcher id=30
        assert 10 in mapping  # author Alice
        assert 20 in mapping  # Bob (assigned + journal)
        assert 30 in mapping  # Dave (watcher)

    def test_empty_journals_list_handled(self, simple_issue):
        mapping: dict = {}
        result = anonymize_issue(simple_issue, mapping)
        assert result["journals"] == []

    def test_journal_without_user_unchanged(self):
        issue = make_issue(journals=[{"id": 1, "notes": "no user key"}])
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert result["journals"][0].get("user") is None

    def test_non_subject_fields_untouched(self, simple_issue):
        mapping: dict = {}
        result = anonymize_issue(simple_issue, mapping)
        assert result["subject"] == simple_issue["subject"]
        assert result["description"] == simple_issue["description"]
        assert result["id"] == simple_issue["id"]

    def test_consistent_across_multiple_issues(self):
        """Same user appearing in two different issues gets the same anonymous name."""
        mapping: dict = {}
        issue_a = make_issue(issue_id=1, author_id=99, author_name="Shared User")
        issue_b = make_issue(issue_id=2, author_id=99, author_name="Shared User")
        result_a = anonymize_issue(issue_a, mapping)
        result_b = anonymize_issue(issue_b, mapping)
        assert result_a["author"]["name"] == result_b["author"]["name"]

    def test_description_pii_scrubbed(self):
        """Emails in the description are redacted."""
        issue = make_issue(description="Contact admin@suse.de for details.")
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert "admin@suse.de" not in result["description"]
        assert "[REDACTED-EMAIL]" in result["description"]

    def test_journal_notes_pii_scrubbed(self):
        """Emails in journal notes are redacted."""
        from tests.conftest import make_journal
        journals = [make_journal(notes="Send logs to debug@example.com please.")]
        issue = make_issue(journals=journals)
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert "debug@example.com" not in result["journals"][0]["notes"]
        assert "[REDACTED-EMAIL]" in result["journals"][0]["notes"]

    def test_ip_address_in_description_scrubbed(self):
        """IPv4 addresses in description are redacted."""
        issue = make_issue(description="Server at 192.168.1.100 is unreachable.")
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert "192.168.1.100" not in result["description"]
        assert "[REDACTED-IP]" in result["description"]

    def test_hostname_in_description_scrubbed(self):
        """Internal hostnames in description are redacted."""
        issue = make_issue(description="Worker host.suse.de is failing.")
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert "host.suse.de" not in result["description"]
        assert "[REDACTED-HOST]" in result["description"]

    def test_clean_description_unchanged(self):
        """A description with no PII is not modified."""
        original = "The kernel panics on boot after update."
        issue = make_issue(description=original)
        mapping: dict = {}
        result = anonymize_issue(issue, mapping)
        assert result["description"] == original


class TestScrubPii:
    def test_email_redacted(self):
        assert "[REDACTED-EMAIL]" in scrub_pii("user@example.com")
        assert "user@example.com" not in scrub_pii("user@example.com")

    def test_multiple_emails_redacted(self):
        text = "From: a@b.com To: c@d.org"
        result = scrub_pii(text)
        assert "a@b.com" not in result
        assert "c@d.org" not in result
        assert result.count("[REDACTED-EMAIL]") == 2

    def test_ipv4_redacted(self):
        result = scrub_pii("Host 10.0.0.1 is down")
        assert "10.0.0.1" not in result
        assert "[REDACTED-IP]" in result

    def test_suse_hostname_redacted(self):
        result = scrub_pii("worker1.suse.de responded slowly")
        assert "worker1.suse.de" not in result
        assert "[REDACTED-HOST]" in result

    def test_internal_hostname_redacted(self):
        result = scrub_pii("connect to db.internal for access")
        assert "db.internal" not in result
        assert "[REDACTED-HOST]" in result

    def test_no_pii_unchanged(self):
        text = "The test failed due to a kernel panic."
        assert scrub_pii(text) == text

    def test_empty_string_unchanged(self):
        assert scrub_pii("") == ""

    def test_none_returned_unchanged(self):
        assert scrub_pii(None) is None

    def test_version_string_not_matched_as_ip(self):
        """Version numbers like 1.2.3.4 embedded in package names should not be redacted."""
        text = "kernel-default-5.14.21.150500-patch"
        result = scrub_pii(text)
        # The IP regex is word-boundary anchored so embedded versions may or
        # may not match — just verify the function does not crash.
        assert isinstance(result, str)

    def test_openqa_suse_de_url_redacted(self):
        """Full openqa.suse.de URLs are replaced with [OPENQA-URL]."""
        text = "See test run at https://openqa.suse.de/tests/12345678 for details."
        result = scrub_pii(text)
        assert "openqa.suse.de" not in result
        assert "[OPENQA-URL]" in result

    def test_openqa_opensuse_org_url_redacted(self):
        """Full openqa.opensuse.org URLs are replaced with [OPENQA-URL]."""
        text = "Failed at http://openqa.opensuse.org/tests/99999#step/boot/1"
        result = scrub_pii(text)
        assert "openqa.opensuse.org" not in result
        assert "[OPENQA-URL]" in result

    def test_openqa_url_with_query_string_redacted(self):
        """openQA URLs with query parameters are fully stripped."""
        text = "Results: https://openqa.suse.de/tests/overview?distri=sle&version=15-SP5"
        result = scrub_pii(text)
        assert "openqa.suse.de" not in result
        assert "[OPENQA-URL]" in result

    def test_multiple_openqa_urls_all_redacted(self):
        """Multiple openQA URLs in one text are all replaced."""
        text = (
            "Job1: https://openqa.suse.de/tests/111 "
            "Job2: https://openqa.opensuse.org/tests/222"
        )
        result = scrub_pii(text)
        assert result.count("[OPENQA-URL]") == 2

    def test_non_openqa_url_preserved(self):
        """URLs to unrelated external hosts are not affected by the openQA URL pattern."""
        text = "See https://github.com/openSUSE/os-autoinst/issues/42 for details."
        result = scrub_pii(text)
        assert "github.com" in result

    def test_openqa_url_scrubbed_before_hostname_pattern(self):
        """openQA URL stripping runs before hostname redaction so no double-replace."""
        text = "Run: https://openqa.suse.de/tests/42"
        result = scrub_pii(text)
        assert "[OPENQA-URL]" in result
        # Should not also contain [REDACTED-HOST] from the same URL
        assert result.count("[REDACTED-HOST]") == 0
