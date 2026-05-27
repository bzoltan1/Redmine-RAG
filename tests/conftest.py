"""
Shared pytest fixtures for the Redmine RAG test suite.
"""

import pytest


# ---------------------------------------------------------------------------
# Synthetic Redmine issue fixtures
# ---------------------------------------------------------------------------

def make_issue(
    issue_id: int = 1,
    subject: str = "Test issue subject",
    description: str = "Test description",
    status: str = "Open",
    priority: str = "Normal",
    tracker: str = "Bug",
    project_name: str = "test-project",
    project_identifier: str = "test-project",
    author_id: int = 10,
    author_name: str = "Alice",
    assigned_id: int | None = None,
    assigned_name: str | None = None,
    journals: list | None = None,
    watchers: list | None = None,
) -> dict:
    """Return a minimal synthetic Redmine issue dict."""
    issue = {
        "id": issue_id,
        "subject": subject,
        "description": description,
        "status": {"name": status},
        "priority": {"name": priority},
        "tracker": {"name": tracker},
        "project": {"name": project_name},
        "project_identifier": project_identifier,
        "created_on": "2024-01-15T10:00:00Z",
        "updated_on": "2024-06-20T14:30:00Z",
        "author": {"id": author_id, "name": author_name},
        "assigned_to": (
            {"id": assigned_id, "name": assigned_name}
            if assigned_id is not None
            else None
        ),
        "journals": journals if journals is not None else [],
        "watchers": watchers if watchers is not None else [],
    }
    return issue


def make_journal(
    journal_id: int = 1,
    user_id: int = 20,
    user_name: str = "Bob",
    notes: str = "This is a comment.",
) -> dict:
    return {
        "id": journal_id,
        "user": {"id": user_id, "name": user_name},
        "notes": notes,
        "created_on": "2024-02-01T09:00:00Z",
        "details": [],
    }


@pytest.fixture
def simple_issue():
    """A minimal issue with no journals or watchers."""
    return make_issue()


@pytest.fixture
def issue_with_journals():
    """An issue that has two journal entries."""
    journals = [
        make_journal(journal_id=1, user_id=20, user_name="Bob", notes="First comment."),
        make_journal(journal_id=2, user_id=21, user_name="Carol", notes="Second comment."),
    ]
    return make_issue(issue_id=2, journals=journals)


@pytest.fixture
def issue_with_watchers():
    """An issue that has watchers."""
    watchers = [
        {"id": 30, "name": "Dave"},
        {"id": 31, "name": "Eve"},
    ]
    return make_issue(issue_id=3, watchers=watchers)


@pytest.fixture
def issue_full():
    """An issue with journals, watchers, and an assignee."""
    journals = [
        make_journal(journal_id=1, user_id=20, user_name="Bob", notes="Investigating."),
    ]
    watchers = [{"id": 30, "name": "Dave"}]
    return make_issue(
        issue_id=4,
        assigned_id=20,
        assigned_name="Bob",
        journals=journals,
        watchers=watchers,
    )


@pytest.fixture
def ten_issues():
    """A list of 10 synthetic issues for integration tests."""
    issues = []
    for i in range(1, 11):
        journals = [make_journal(journal_id=i * 10, notes=f"Comment on issue {i}.")]
        issues.append(
            make_issue(
                issue_id=i,
                subject=f"Issue number {i}",
                description=f"Description of issue {i}. Relates to kernel networking stack.",
                project_identifier="test-project",
                journals=journals,
            )
        )
    return issues
