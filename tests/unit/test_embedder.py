"""
Unit tests for core/embedder.py

The Ollama client is mocked so no running Ollama server is required.
"""

import pytest
from unittest.mock import MagicMock, patch

from core.embedder import OllamaEmbedder


FAKE_VECTOR = [0.1, 0.2, 0.3]


def make_embedder(model: str = "nomic-embed-text") -> OllamaEmbedder:
    with patch("core.embedder.Client"):
        embedder = OllamaEmbedder(model=model, host="http://localhost:11434")
    return embedder


def mock_embed_response(vectors: list[list[float]]) -> dict:
    return {"embeddings": vectors}


class TestOllamaEmbedderInit:
    def test_model_stored(self):
        embedder = make_embedder(model="mxbai-embed-large")
        assert embedder.model == "mxbai-embed-large"

    def test_name_includes_model(self):
        embedder = make_embedder(model="nomic-embed-text")
        assert "nomic-embed-text" in embedder.name()


class TestEmbed:
    def test_returns_list_of_vectors(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR, FAKE_VECTOR])
        )
        result = embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert result[0] == FAKE_VECTOR

    def test_empty_input_returns_empty_list(self):
        embedder = make_embedder()
        result = embedder.embed([])
        assert result == []

    def test_calls_client_with_correct_model(self):
        embedder = make_embedder(model="nomic-embed-text")
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR])
        )
        embedder.embed(["test"])
        # embed() sends the full list in one batch call
        embedder._client.embed.assert_called_once_with(
            model="nomic-embed-text", input=["test"]
        )

    def test_raises_on_missing_embeddings_key(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(return_value={"something_else": []})
        with pytest.raises(ValueError, match="embeddings"):
            embedder.embed(["test"])

    def test_single_text_single_vector(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR])
        )
        result = embedder.embed(["one"])
        assert len(result) == 1


class TestEmbedOne:
    def test_returns_single_vector(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR])
        )
        result = embedder.embed_one("single query")
        assert result == FAKE_VECTOR

    def test_raises_if_no_vectors_returned(self):
        embedder = make_embedder()
        # Empty embeddings list triggers the guard inside embed()
        embedder._client.embed = MagicMock(return_value={"embeddings": []})
        with pytest.raises(ValueError):
            embedder.embed_one("query")


class TestCallProtocol:
    """OllamaEmbedder must satisfy ChromaDB's embedding_function protocol."""

    def test_call_with_list(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR])
        )
        result = embedder(["text"])
        assert result == [FAKE_VECTOR]

    def test_call_with_string(self):
        embedder = make_embedder()
        embedder._client.embed = MagicMock(
            return_value=mock_embed_response([FAKE_VECTOR])
        )
        result = embedder("single string")
        assert result == [FAKE_VECTOR]
