"""
core/redmine_client.py — Redmine REST API client.

Handles pagination, per-issue fetching, HTTP error classification, and
adaptive rate limiting with exponential backoff.

Rate limiting strategy:
  - Base delay starts at `rate_limit` seconds (default 0.5s).
  - On HTTP 429 / 503 the delay doubles (capped at `backoff_max`).
  - On any successful response the delay resets to `rate_limit`.
  - Network errors (DNS, timeout) use a separate fixed `retry_delay`.

Returns plain Python dicts. No file I/O.
"""

import time
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Sentinel returned by fetch_issue on unrecoverable HTTP errors
# (404, 403, 401) so callers can distinguish "gone" from "retry".
ISSUE_NOT_FOUND = object()

# HTTP status codes that indicate server-side rate limiting or overload.
_BACKOFF_STATUSES = {429, 503}


class RedmineClient:
    """
    Thin wrapper around the Redmine JSON REST API.

    Parameters
    ----------
    base_url:    Redmine instance URL (no trailing slash).
    api_key:     Redmine API key.
    rate_limit:  Base inter-request delay in seconds (default 0.5).
                 This is the floor — the actual delay increases under
                 server pressure and resets to this value on success.
    backoff_max: Maximum inter-request delay in seconds (default 60).
    timeout:     HTTP request timeout in seconds.
    max_retries: Number of retry attempts on transient errors.
    retry_delay: Fixed delay in seconds after a network-level error.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        rate_limit: float = 0.5,
        backoff_max: float = 60.0,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.backoff_max = backoff_max
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Current adaptive delay — starts at floor, grows/shrinks at runtime.
        self._current_delay: float = rate_limit

        self._session = requests.Session()
        self._session.headers.update(
            {"X-Redmine-API-Key": api_key, "Content-Type": "application/json"}
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_total_count(self, project_id: str) -> int:
        """
        Return the total number of issues in *project_id*.
        Raises RuntimeError if the API call fails.
        """
        data = self._get(
            f"{self.base_url}/issues.json",
            params={"project_id": project_id, "limit": 1, "status_id": "*"},
        )
        return data.get("total_count", 0)

    def fetch_issues_page(
        self,
        project_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Fetch one page of issues for *project_id*.

        Returns
        -------
        (issues, total_count) where *issues* is a list of issue dicts
        and *total_count* is the total number of issues in the project.
        """
        data = self._get(
            f"{self.base_url}/issues.json",
            params={
                "project_id": project_id,
                "limit": limit,
                "offset": offset,
                "include": "journals",
                "status_id": "*",
                "sort": "id:asc",
            },
        )
        issues = data.get("issues", [])
        for issue in issues:
            issue["project_identifier"] = project_id
        return issues, data.get("total_count", 0)

    def fetch_issue(self, issue_id: int) -> dict[str, Any] | object:
        """
        Fetch a single issue with full journal/watcher data.

        Returns
        -------
        The issue dict on success.
        ISSUE_NOT_FOUND sentinel if the issue is gone (404/403/401).
        Raises requests.RequestException after all retries are exhausted.
        """
        url = f"{self.base_url}/issues/{issue_id}.json"
        params = {"include": "journals,children,attachments,relations,watchers"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning(
                    "Issue #%d: network error on attempt %d/%d: %s",
                    issue_id, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise

            if resp.status_code in (401, 403, 404):
                logger.warning(
                    "Issue #%d: HTTP %d — treating as not found.",
                    issue_id, resp.status_code,
                )
                return ISSUE_NOT_FOUND

            if resp.status_code in _BACKOFF_STATUSES:
                wait = self._backoff()
                logger.warning(
                    "Issue #%d: HTTP %d — backing off %.1fs (attempt %d/%d).",
                    issue_id, resp.status_code, wait, attempt, self.max_retries,
                )
                time.sleep(wait)
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                logger.warning(
                    "Issue #%d: HTTP error on attempt %d/%d: %s",
                    issue_id, attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                raise

            self._reset_delay()
            time.sleep(self._current_delay)
            return resp.json().get("issue", {})

        return ISSUE_NOT_FOUND  # pragma: no cover

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _backoff(self) -> float:
        """Double the current delay (capped at backoff_max) and return it."""
        self._current_delay = min(self._current_delay * 2, self.backoff_max)
        logger.debug("Adaptive delay increased to %.1fs.", self._current_delay)
        return self._current_delay

    def _reset_delay(self) -> None:
        """Reset the current delay to the base rate_limit floor."""
        if self._current_delay != self.rate_limit:
            logger.debug(
                "Adaptive delay reset to %.1fs.", self.rate_limit
            )
            self._current_delay = self.rate_limit

    def _get(self, url: str, params: dict | None = None) -> dict[str, Any]:
        """
        Perform a GET request with adaptive backoff on 429/503 and retry
        on transient network errors. Raises RuntimeError on persistent failure.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)

                if resp.status_code in _BACKOFF_STATUSES:
                    wait = self._backoff()
                    logger.warning(
                        "GET %s: HTTP %d — backing off %.1fs (attempt %d/%d).",
                        url, resp.status_code, wait, attempt, self.max_retries,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                self._reset_delay()
                time.sleep(self._current_delay)
                return resp.json()

            except requests.RequestException as exc:
                logger.warning(
                    "GET %s attempt %d/%d failed: %s", url, attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to GET {url} after {self.max_retries} attempts"
                    ) from exc
        return {}  # pragma: no cover
