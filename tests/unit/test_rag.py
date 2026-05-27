"""
Unit tests for core/rag.py

All external I/O (Ollama, VectorStore) is mocked so no running server or
ChromaDB instance is required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from core import rag
from core.rag import (
    extract_filters,
    retrieve,
    build_prompt,
    generate,
    answer,
    _KNOWN_STATUSES,
    _KNOWN_PRIORITIES,
)
from core.store import VectorStore


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_result(
    issue_id: str = "42",
    project: str = "containers",
    status: str = "Open",
    priority: str = "High",
    text: str = "Subject: Container fails\nDescription: timeout on startup",
    score: float = 0.12,
    created_on: str = "2024-01-15T10:00:00Z",
    updated_on: str = "2024-06-20T14:30:00Z",
) -> dict:
    return {
        "id": f"issue_{issue_id}",
        "text": text,
        "metadata": {
            "issue_id":   issue_id,
            "project":    project,
            "status":     status,
            "priority":   priority,
            "subject":    "Container fails",
            "created_on": created_on,
            "updated_on": updated_on,
        },
        "score": score,
    }


def make_store_mock(results: list[dict] | None = None) -> MagicMock:
    store = MagicMock(spec=VectorStore)
    store.query.return_value = results if results is not None else [make_result()]
    return store


def mock_chat_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------------------------------------------------------------------------
# extract_filters()
# ---------------------------------------------------------------------------

class TestExtractFilters:
    def test_returns_dict(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{"status": "Rejected"}')
            result = extract_filters("Show rejected issues", model="llama3")
        assert isinstance(result, dict)

    def test_extracts_known_status(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{"status": "Rejected"}')
            result = extract_filters("Show rejected issues", model="llama3")
        assert result.get("status") == "Rejected"

    def test_extracts_known_priority(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{"priority": "High"}')
            result = extract_filters("Show high priority bugs", model="llama3")
        assert result.get("priority") == "High"

    def test_unknown_status_filtered_out(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{"status": "Imaginary"}')
            result = extract_filters("anything", model="llama3")
        assert "status" not in result

    def test_unknown_priority_filtered_out(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{"priority": "Critical"}')
            result = extract_filters("anything", model="llama3")
        assert "priority" not in result

    def test_empty_json_returns_empty_dict(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response('{}')
            result = extract_filters("generic question", model="llama3")
        assert result == {}

    def test_api_error_returns_empty_dict(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.side_effect = RuntimeError("connection refused")
            result = extract_filters("anything", model="llama3")
        assert result == {}

    def test_invalid_json_returns_empty_dict(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response("not json at all")
            result = extract_filters("anything", model="llama3")
        assert result == {}

    def test_markdown_fenced_json_parsed(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response(
                '```json\n{"status": "Resolved"}\n```'
            )
            result = extract_filters("resolved issues", model="llama3")
        assert result.get("status") == "Resolved"

    def test_both_filters_extracted(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response(
                '{"status": "New", "priority": "Urgent"}'
            )
            result = extract_filters("urgent new issues", model="llama3")
        assert result.get("status") == "New"
        assert result.get("priority") == "Urgent"


# ---------------------------------------------------------------------------
# retrieve()
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_delegates_to_store_query(self):
        store = make_store_mock([make_result()])
        results = retrieve("why does container fail?", store, top_k=3)
        store.query.assert_called_once_with("why does container fail?", top_k=3, where=None, deduplicate=True)
        assert len(results) == 1

    def test_returns_empty_list_on_no_results(self):
        store = make_store_mock([])
        results = retrieve("anything", store)
        assert results == []

    def test_where_filter_forwarded(self):
        store = make_store_mock()
        retrieve("question", store, where={"status": {"$eq": "Rejected"}})
        _, kwargs = store.query.call_args
        assert kwargs["where"] == {"status": {"$eq": "Rejected"}}


# ---------------------------------------------------------------------------
# build_prompt()
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_question_in_prompt(self):
        prompt = build_prompt("Why does container fail?", [])
        assert "Why does container fail?" in prompt

    def test_no_results_fallback_message(self):
        prompt = build_prompt("question", [])
        assert "no issues retrieved" in prompt

    def test_issue_id_in_prompt(self):
        prompt = build_prompt("question", [make_result(issue_id="99")])
        assert "#99" in prompt

    def test_project_in_prompt(self):
        prompt = build_prompt("question", [make_result(project="qe-kernel")])
        assert "qe-kernel" in prompt

    def test_status_in_prompt(self):
        prompt = build_prompt("question", [make_result(status="Resolved")])
        assert "Resolved" in prompt

    def test_priority_in_prompt(self):
        prompt = build_prompt("question", [make_result(priority="Urgent")])
        assert "Urgent" in prompt

    def test_created_on_in_prompt(self):
        prompt = build_prompt("question", [make_result(created_on="2024-01-15T10:00:00Z")])
        assert "2024-01-15T10:00:00Z" in prompt

    def test_updated_on_in_prompt(self):
        prompt = build_prompt("question", [make_result(updated_on="2024-06-20T14:30:00Z")])
        assert "2024-06-20T14:30:00Z" in prompt

    def test_issue_text_in_prompt(self):
        prompt = build_prompt("question", [make_result(text="Kernel oops on boot")])
        assert "Kernel oops on boot" in prompt

    def test_multiple_issues_separated(self):
        results = [make_result(issue_id="1"), make_result(issue_id="2")]
        prompt = build_prompt("question", results)
        assert "#1" in prompt and "#2" in prompt

    def test_missing_metadata_uses_placeholder(self):
        result = {"id": "issue_5", "text": "Some text", "metadata": {}, "score": 0.1}
        prompt = build_prompt("question", [result])
        assert "?" in prompt


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_answer_string(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response("The answer.")
            result = generate("some prompt", model="llama3")
        assert result == "The answer."

    def test_calls_chat_with_correct_model(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response("ok")
            generate("prompt", model="mistral")
            call_kwargs = MockClient.return_value.chat.call_args[1]
            assert call_kwargs["model"] == "mistral"

    def test_system_prompt_in_messages(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response("ok")
            generate("user prompt", model="llama3")
            messages = MockClient.return_value.chat.call_args[1]["messages"]
            roles = [m["role"] for m in messages]
            assert "system" in roles and "user" in roles

    def test_dict_response_format_supported(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = {"message": {"content": "dict-style"}}
            result = generate("prompt", model="llama3")
        assert result == "dict-style"

    def test_host_forwarded_to_client(self):
        with patch("core.rag.Client") as MockClient:
            MockClient.return_value.chat.return_value = mock_chat_response("ok")
            generate("prompt", model="llama3", host="http://custom:11434")
            MockClient.assert_called_once_with(host="http://custom:11434")


# ---------------------------------------------------------------------------
# answer()
# ---------------------------------------------------------------------------

class TestAnswer:
    def _mock_generate(self, text: str = "Generated answer."):
        return patch("core.rag.generate", return_value=text)

    def _mock_extract(self, filters: dict = {}):
        return patch("core.rag.extract_filters", return_value=filters)

    def test_returns_three_tuple(self):
        store = make_store_mock([make_result()])
        with self._mock_generate(), self._mock_extract():
            result = answer("question", store, chat_model="llama3")
        assert isinstance(result, tuple) and len(result) == 3

    def test_answer_text_returned(self):
        store = make_store_mock([make_result()])
        with self._mock_generate("My answer"), self._mock_extract():
            text, _, _ = answer("question", store, chat_model="llama3")
        assert text == "My answer"

    def test_retrieved_issues_returned(self):
        issues = [make_result(issue_id="10"), make_result(issue_id="11")]
        store = make_store_mock(issues)
        with self._mock_generate(), self._mock_extract():
            _, retrieved, _ = answer("question", store, chat_model="llama3")
        assert len(retrieved) == 2

    def test_filters_returned(self):
        store = make_store_mock()
        with self._mock_generate(), self._mock_extract({"status": "Rejected"}):
            _, _, filters = answer("question", store, chat_model="llama3")
        assert filters == {"status": "Rejected"}

    def test_single_filter_builds_correct_where_clause(self):
        store = make_store_mock()
        with self._mock_generate(), self._mock_extract({"status": "Rejected"}):
            answer("show rejected", store, chat_model="llama3")
        _, kwargs = store.query.call_args
        assert kwargs["where"] == {"status": {"$eq": "Rejected"}}

    def test_two_filters_build_and_clause(self):
        store = make_store_mock()
        with self._mock_generate(), self._mock_extract({"status": "New", "priority": "High"}):
            answer("question", store, chat_model="llama3")
        _, kwargs = store.query.call_args
        assert "$and" in kwargs["where"]

    def test_no_filters_no_where_clause(self):
        store = make_store_mock()
        with self._mock_generate(), self._mock_extract({}):
            answer("question", store, chat_model="llama3")
        _, kwargs = store.query.call_args
        assert kwargs["where"] is None

    def test_filter_extraction_skipped_when_disabled(self):
        store = make_store_mock()
        with self._mock_generate():
            with patch("core.rag.extract_filters") as mock_ext:
                answer("question", store, chat_model="llama3", extract_metadata_filters=False)
                mock_ext.assert_not_called()

    def test_empty_filter_results_retried_without_filter(self):
        """If filters produce no results, the query is retried without them."""
        store = MagicMock(spec=VectorStore)
        # First call (with filter) returns nothing; second call returns results
        store.query.side_effect = [[], [make_result()]]
        with self._mock_generate(), self._mock_extract({"status": "Rejected"}):
            text, retrieved, filters = answer("question", store, chat_model="llama3")
        assert store.query.call_count == 2
        assert filters == {}  # cleared after fallback

    def test_top_k_forwarded(self):
        store = make_store_mock()
        with self._mock_generate(), self._mock_extract():
            answer("question", store, chat_model="llama3", top_k=8)
        _, kwargs = store.query.call_args
        assert kwargs["top_k"] == 8
