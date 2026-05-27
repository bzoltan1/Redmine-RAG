"""
Unit tests for core/redmine_client.py

All HTTP calls are intercepted with the `responses` library so no real
network traffic is made.
"""

import pytest
import responses as rsps_lib
from requests.exceptions import ConnectionError as RequestsConnectionError

from core.redmine_client import RedmineClient, ISSUE_NOT_FOUND


BASE = "https://redmine.example.com"


def make_client(**kwargs) -> RedmineClient:
    defaults = dict(
        base_url=BASE,
        api_key="test-key",
        rate_limit=0,        # no sleeping in tests
        backoff_max=0,       # no sleeping in tests
        timeout=5,
        max_retries=2,
        retry_delay=0,       # no sleeping in tests
    )
    defaults.update(kwargs)
    return RedmineClient(**defaults)


def issues_response(issues: list, total: int) -> dict:
    return {"issues": issues, "total_count": total, "offset": 0, "limit": 100}


def single_issue_response(issue_id: int) -> dict:
    return {
        "issue": {
            "id": issue_id,
            "subject": f"Issue {issue_id}",
            "journals": [{"id": 1, "notes": "A note", "user": {"id": 5, "name": "Tester"}}],
        }
    }


# ---------------------------------------------------------------------------
# get_total_count
# ---------------------------------------------------------------------------

class TestGetTotalCount:
    @rsps_lib.activate
    def test_returns_total_count(self):
        rsps_lib.add(
            rsps_lib.GET,
            f"{BASE}/issues.json",
            json={"issues": [], "total_count": 42},
            status=200,
        )
        client = make_client()
        assert client.get_total_count("myproject") == 42

    @rsps_lib.activate
    def test_raises_on_persistent_failure(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", body=RequestsConnectionError("DNS fail"))
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", body=RequestsConnectionError("DNS fail"))
        client = make_client()
        with pytest.raises(RuntimeError, match="Failed to GET"):
            client.get_total_count("myproject")


# ---------------------------------------------------------------------------
# fetch_issues_page
# ---------------------------------------------------------------------------

class TestFetchIssuesPage:
    @rsps_lib.activate
    def test_returns_issues_and_total(self):
        issues = [{"id": 1, "subject": "A"}, {"id": 2, "subject": "B"}]
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", json=issues_response(issues, 50))
        client = make_client()
        result_issues, total = client.fetch_issues_page("proj", offset=0)
        assert total == 50
        assert len(result_issues) == 2

    @rsps_lib.activate
    def test_project_identifier_injected(self):
        issues = [{"id": 1, "subject": "A"}]
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", json=issues_response(issues, 1))
        client = make_client()
        result_issues, _ = client.fetch_issues_page("my-project")
        assert result_issues[0]["project_identifier"] == "my-project"

    @rsps_lib.activate
    def test_empty_project_returns_empty_list(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", json=issues_response([], 0))
        client = make_client()
        result_issues, total = client.fetch_issues_page("empty")
        assert result_issues == []
        assert total == 0

    @rsps_lib.activate
    def test_offset_forwarded_in_request(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", json=issues_response([], 0))
        client = make_client()
        client.fetch_issues_page("proj", offset=200)
        assert "offset=200" in rsps_lib.calls[0].request.url


# ---------------------------------------------------------------------------
# fetch_issue
# ---------------------------------------------------------------------------

class TestFetchIssue:
    @rsps_lib.activate
    def test_returns_issue_dict(self):
        rsps_lib.add(
            rsps_lib.GET,
            f"{BASE}/issues/42.json",
            json=single_issue_response(42),
            status=200,
        )
        client = make_client()
        issue = client.fetch_issue(42)
        assert issue["id"] == 42

    @rsps_lib.activate
    def test_returns_sentinel_on_404(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/99.json", status=404)
        client = make_client()
        assert client.fetch_issue(99) is ISSUE_NOT_FOUND

    @rsps_lib.activate
    def test_returns_sentinel_on_403(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/99.json", status=403)
        client = make_client()
        assert client.fetch_issue(99) is ISSUE_NOT_FOUND

    @rsps_lib.activate
    def test_returns_sentinel_on_401(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/99.json", status=401)
        client = make_client()
        assert client.fetch_issue(99) is ISSUE_NOT_FOUND

    @rsps_lib.activate
    def test_retries_on_transient_error_then_succeeds(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/7.json", body=RequestsConnectionError("DNS"))
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/7.json", json=single_issue_response(7), status=200)
        client = make_client(max_retries=2)
        issue = client.fetch_issue(7)
        assert issue["id"] == 7

    @rsps_lib.activate
    def test_raises_after_all_retries_exhausted(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/7.json", body=RequestsConnectionError("DNS"))
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/7.json", body=RequestsConnectionError("DNS"))
        client = make_client(max_retries=2)
        with pytest.raises(RequestsConnectionError):
            client.fetch_issue(7)

    @rsps_lib.activate
    def test_journals_included_in_params(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/1.json", json=single_issue_response(1))
        client = make_client()
        client.fetch_issue(1)
        assert "journals" in rsps_lib.calls[0].request.url

    @rsps_lib.activate
    def test_api_key_in_headers(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/1.json", json=single_issue_response(1))
        client = make_client(api_key="secret-key")
        client.fetch_issue(1)
        assert rsps_lib.calls[0].request.headers["X-Redmine-API-Key"] == "secret-key"


# ---------------------------------------------------------------------------
# Adaptive backoff
# ---------------------------------------------------------------------------

class TestAdaptiveBackoff:
    def test_delay_doubles_on_backoff(self):
        client = make_client(rate_limit=1.0, backoff_max=60.0)
        assert client._current_delay == 1.0
        client._backoff()
        assert client._current_delay == 2.0
        client._backoff()
        assert client._current_delay == 4.0

    def test_delay_capped_at_backoff_max(self):
        client = make_client(rate_limit=1.0, backoff_max=5.0)
        for _ in range(10):
            client._backoff()
        assert client._current_delay == 5.0

    def test_delay_resets_after_success(self):
        client = make_client(rate_limit=0.5, backoff_max=60.0)
        client._backoff()
        client._backoff()
        assert client._current_delay == 2.0
        client._reset_delay()
        assert client._current_delay == 0.5

    @rsps_lib.activate
    def test_fetch_issue_retries_on_429(self):
        """429 triggers a backoff retry, then succeeds."""
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/5.json", status=429)
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/5.json",
                     json=single_issue_response(5), status=200)
        client = make_client(max_retries=2)
        issue = client.fetch_issue(5)
        assert issue["id"] == 5

    @rsps_lib.activate
    def test_fetch_issue_retries_on_503(self):
        """503 triggers a backoff retry, then succeeds."""
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/6.json", status=503)
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/6.json",
                     json=single_issue_response(6), status=200)
        client = make_client(max_retries=2)
        issue = client.fetch_issue(6)
        assert issue["id"] == 6

    @rsps_lib.activate
    def test_get_retries_on_429(self):
        """_get() backs off and retries on 429."""
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json", status=429)
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json={"issues": [], "total_count": 0}, status=200)
        client = make_client(max_retries=2)
        result = client._get(f"{BASE}/issues.json")
        assert result["total_count"] == 0

    @rsps_lib.activate
    def test_delay_resets_after_successful_fetch(self):
        """After a 429 retry succeeds, delay resets to rate_limit floor."""
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/8.json", status=429)
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues/8.json",
                     json=single_issue_response(8), status=200)
        client = make_client(max_retries=2, rate_limit=0.0)
        client.fetch_issue(8)
        assert client._current_delay == 0.0


# ---------------------------------------------------------------------------
# fetch_updated_since (incremental sync)
# ---------------------------------------------------------------------------

class TestFetchUpdatedSince:
    @rsps_lib.activate
    def test_returns_issues_and_total(self):
        issues = [{"id": 10, "subject": "New issue", "updated_on": "2026-01-10T00:00:00Z"}]
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response(issues, 1))
        client = make_client()
        result, total = client.fetch_updated_since("proj", since="2025-12-02")
        assert total == 1
        assert len(result) == 1

    @rsps_lib.activate
    def test_updated_on_filter_in_request(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response([], 0))
        client = make_client()
        client.fetch_updated_since("proj", since="2025-12-02")
        url = rsps_lib.calls[0].request.url
        assert "updated_on" in url
        assert "2025-12-02" in url

    @rsps_lib.activate
    def test_project_identifier_injected(self):
        issues = [{"id": 5, "subject": "Updated"}]
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response(issues, 1))
        client = make_client()
        result, _ = client.fetch_updated_since("my-proj", since="2025-12-02")
        assert result[0]["project_identifier"] == "my-proj"

    @rsps_lib.activate
    def test_journals_included_in_params(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response([], 0))
        client = make_client()
        client.fetch_updated_since("proj", since="2025-12-02")
        assert "journals" in rsps_lib.calls[0].request.url

    @rsps_lib.activate
    def test_empty_result_when_no_updates(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response([], 0))
        client = make_client()
        result, total = client.fetch_updated_since("proj", since="2099-01-01")
        assert result == []
        assert total == 0

    @rsps_lib.activate
    def test_offset_forwarded(self):
        rsps_lib.add(rsps_lib.GET, f"{BASE}/issues.json",
                     json=issues_response([], 0))
        client = make_client()
        client.fetch_updated_since("proj", since="2025-12-02", offset=100)
        assert "offset=100" in rsps_lib.calls[0].request.url
