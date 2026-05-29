"""
Unit tests for core/store.py

Uses chromadb.EphemeralClient (in-memory) via VectorStore(db_path=None)
so no disk access or Ollama server is required. The embedder is stubbed
to return deterministic fixed-dimension vectors.
"""

import uuid
import pytest
from unittest.mock import MagicMock

from core.store import VectorStore, _deduplicate_by_parent
from core.embedder import OllamaEmbedder


DIM = 4  # small fixed embedding dimension for tests


class StubEmbedder:
    """
    A real (non-mock) embedder stub that returns deterministic float vectors.
    Used in store tests so that ChromaDB receives proper numpy-compatible data.
    """

    model = "stub-model"

    def name(self) -> str:
        return "stub-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(i) / DIM for i in range(DIM)] for _ in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed(input)


def fake_embedder() -> StubEmbedder:
    return StubEmbedder()


def make_store(collection_name: str | None = None, batch_size: int = 10) -> VectorStore:
    # Use a unique collection name per call so tests don't share state via
    # the EphemeralClient's in-memory store.
    name = collection_name or f"test_{uuid.uuid4().hex}"
    return VectorStore(
        db_path=None,
        collection_name=name,
        embedder=fake_embedder(),
        batch_size=batch_size,
    )


def make_doc(doc_id: str = "issue_1", text: str = "test text") -> dict:
    return {
        "id": doc_id,
        "text": text,
        "metadata": {"issue_id": doc_id.replace("issue_", ""), "subject": text},
    }


class TestVectorStoreInit:
    def test_ephemeral_client_when_db_path_none(self):
        store = make_store()
        assert store.count() == 0

    def test_ephemeral_client_when_db_path_memory(self):
        embedder = fake_embedder()
        store = VectorStore(db_path=":memory:", collection_name=f"col_{uuid.uuid4().hex}", embedder=embedder)
        assert store.count() == 0


class TestCount:
    def test_zero_on_empty_collection(self):
        store = make_store()
        assert store.count() == 0

    def test_count_after_add(self):
        store = make_store()
        store.add([make_doc("issue_1"), make_doc("issue_2")])
        assert store.count() == 2


class TestAdd:
    def test_single_document_added(self):
        store = make_store()
        store.add([make_doc("issue_1", "kernel crash on boot")])
        assert store.count() == 1

    def test_multiple_documents_added(self):
        store = make_store()
        docs = [make_doc(f"issue_{i}", f"text {i}") for i in range(5)]
        store.add(docs)
        assert store.count() == 5

    def test_batching_respected(self):
        """With batch_size=2 and 5 docs, we expect 3 upsert calls."""
        store = VectorStore(db_path=None, collection_name=f"col_{uuid.uuid4().hex}", embedder=fake_embedder(), batch_size=2)
        original_upsert = store._collection.upsert
        call_count = []

        def counting_upsert(**kwargs):
            call_count.append(1)
            return original_upsert(**kwargs)

        store._collection.upsert = counting_upsert
        docs = [make_doc(f"issue_{i}", f"text {i}") for i in range(5)]
        store.add(docs)
        assert sum(call_count) == 3  # ceil(5/2) = 3

    def test_empty_list_no_error(self):
        store = make_store()
        store.add([])
        assert store.count() == 0


class TestReset:
    def test_reset_clears_documents(self):
        store = make_store()
        store.add([make_doc("issue_1"), make_doc("issue_2")])
        assert store.count() == 2
        store.reset()
        assert store.count() == 0

    def test_reset_on_empty_collection_no_error(self):
        store = make_store()
        store.reset()  # Should not raise
        assert store.count() == 0


class TestQuery:
    def test_returns_list(self):
        store = make_store()
        store.add([make_doc("issue_1", "kernel networking failure")])
        results = store.query("kernel crash", top_k=1)
        assert isinstance(results, list)

    def test_result_has_expected_keys(self):
        store = make_store()
        store.add([make_doc("issue_1", "container startup timeout")])
        results = store.query("container issue", top_k=1)
        assert len(results) == 1
        result = results[0]
        assert "id" in result
        assert "text" in result
        assert "metadata" in result
        assert "score" in result

    def test_top_k_limits_results(self):
        store = make_store()
        docs = [make_doc(f"issue_{i}", f"text about topic {i}") for i in range(10)]
        store.add(docs)
        results = store.query("topic", top_k=3)
        assert len(results) == 3

    def test_top_k_capped_by_collection_size(self):
        store = make_store()
        store.add([make_doc("issue_1", "only doc")])
        results = store.query("anything", top_k=5)
        assert len(results) == 1

    def test_result_id_matches_inserted(self):
        store = make_store()
        store.add([make_doc("issue_42", "specific text")])
        results = store.query("specific", top_k=1)
        assert results[0]["id"] == "issue_42"

    def test_score_is_numeric(self):
        store = make_store()
        store.add([make_doc("issue_1", "text")])
        results = store.query("text", top_k=1)
        assert isinstance(results[0]["score"], (int, float))

    def test_deduplicate_false_returns_all_chunks(self):
        """With deduplicate=False, multiple chunks from the same issue are returned."""
        store = make_store()
        # Two chunks from issue_42
        store.add([
            {"id": "issue_42",         "text": "description text", "metadata": {"issue_id": "42", "subject": "s"}},
            {"id": "issue_42_chunk_1", "text": "journal chunk",    "metadata": {"issue_id": "42", "subject": "s"}},
            {"id": "issue_99",         "text": "other issue",      "metadata": {"issue_id": "99", "subject": "s"}},
        ])
        results = store.query("text", top_k=5, deduplicate=False)
        issue_ids = [r["metadata"]["issue_id"] for r in results]
        # Without dedup, both chunks from 42 can appear
        assert issue_ids.count("42") >= 1


class TestDeduplicateByParent:
    def _hit(self, issue_id: str, chunk_id: str, score: float) -> dict:
        return {
            "id": chunk_id,
            "text": f"text for {chunk_id}",
            "metadata": {"issue_id": issue_id},
            "score": score,
        }

    def test_single_chunk_per_issue_unchanged(self):
        hits = [self._hit("1", "issue_1", 0.1), self._hit("2", "issue_2", 0.2)]
        result = _deduplicate_by_parent(hits)
        assert len(result) == 2

    def test_best_chunk_kept_per_issue(self):
        hits = [
            self._hit("42", "issue_42",         0.3),
            self._hit("42", "issue_42_chunk_1", 0.1),  # better score
            self._hit("42", "issue_42_chunk_2", 0.5),
        ]
        result = _deduplicate_by_parent(hits)
        assert len(result) == 1
        assert result[0]["id"] == "issue_42_chunk_1"
        assert result[0]["score"] == 0.1

    def test_results_sorted_by_score(self):
        hits = [
            self._hit("1", "issue_1", 0.5),
            self._hit("2", "issue_2", 0.1),
            self._hit("3", "issue_3", 0.3),
        ]
        result = _deduplicate_by_parent(hits)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores)

    def test_empty_input_returns_empty(self):
        assert _deduplicate_by_parent([]) == []

    def test_missing_metadata_uses_chunk_id_as_parent(self):
        hits = [{"id": "issue_1", "text": "t", "metadata": {}, "score": 0.1}]
        result = _deduplicate_by_parent(hits)
        assert len(result) == 1
