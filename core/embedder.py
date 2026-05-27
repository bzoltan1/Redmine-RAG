"""
core/embedder.py — Ollama embedding wrapper.

Single, consistent implementation used at both ingest time and query time.
Eliminates the ollama.Client vs langchain_ollama mismatch in the original code.
"""

import logging
from typing import Any

from ollama import Client

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Wraps the Ollama Python SDK for text embedding.

    Parameters
    ----------
    model:         Ollama model name (e.g. "nomic-embed-text", "mxbai-embed-large").
    host:          Ollama server URL (default: http://localhost:11434).
    max_chars:     Hard character limit applied to each text before sending to
                   Ollama. Prevents context-window errors on documents that are
                   token-dense (e.g. long code blocks, repeated characters).
                   Default 1500 chars, conservatively safe for both
                   nomic-embed-text and mxbai-embed-large (512-token windows).
                   Handles token-dense content (code blocks, repeated chars).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        max_chars: int = 1500,
    ) -> None:
        self.model = model
        self.max_chars = max_chars
        self._client = Client(host=host)

    # ------------------------------------------------------------------
    # ChromaDB embedding function protocol
    # ChromaDB calls the embedding function with a list of strings and
    # expects a list of float lists back.
    # ------------------------------------------------------------------

    def __call__(self, input: list[str] | str) -> list[list[float]]:
        """Make OllamaEmbedder directly usable as a ChromaDB embedding_function."""
        return self.embed(input if isinstance(input, list) else [input])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Each text is embedded individually to avoid exceeding the model's
        context window when batching many long documents in one API call.

        Parameters
        ----------
        texts: List of strings to embed.

        Returns
        -------
        List of embedding vectors (one per input text).
        """
        if not texts:
            return []

        # Truncate all texts to max_chars before sending
        truncated = [t[: self.max_chars] if self.max_chars and len(t) > self.max_chars else t for t in texts]

        # Send as a single batch call to minimise HTTP round-trips.
        # Ollama processes inputs sequentially on the GPU regardless, so
        # batching only saves HTTP overhead — but that overhead adds up.
        response: Any = self._client.embed(model=self.model, input=truncated)
        embeddings = (
            response.get("embeddings")
            if isinstance(response, dict)
            else getattr(response, "embeddings", None)
        )
        if embeddings is None:
            raise ValueError(
                f"Ollama embed response did not contain 'embeddings'. "
                f"Response keys: {list(response.keys()) if isinstance(response, dict) else type(response)}"
            )
        if not embeddings:
            raise ValueError("Ollama returned no embeddings for the input text.")
        return list(embeddings)

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single string and return its vector.

        Convenience wrapper around embed() for the common query case.
        """
        vectors = self.embed([text])
        if not vectors:
            raise ValueError("Ollama returned no embeddings for the input text.")
        return vectors[0]

    # ------------------------------------------------------------------
    # ChromaDB EmbeddingFunction compatibility helpers
    # ------------------------------------------------------------------

    def name(self) -> str:
        return f"ollama-{self.model}"
