"""
core/store.py — ChromaDB vector store wrapper.

Provides a clean interface for adding documents and querying by semantic
similarity. Uses OllamaEmbedder for all embedding operations to ensure
consistency between ingest and query time.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from core.embedder import OllamaEmbedder

logger = logging.getLogger(__name__)


def _deduplicate_by_parent(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep only the best-scoring chunk per parent issue_id.

    Chunks from prepare_chunks() share the same issue_id in metadata but
    have different ChromaDB document IDs (e.g. issue_42_chunk_1). This
    function ensures each parent issue appears at most once in results,
    represented by its lowest-distance (most relevant) chunk.

    ChromaDB returns L2 distances so lower = more similar.
    """
    seen: dict[str, dict[str, Any]] = {}
    for hit in hits:
        parent_id = (hit.get("metadata") or {}).get("issue_id", hit["id"])
        if parent_id not in seen or hit["score"] < seen[parent_id]["score"]:
            seen[parent_id] = hit
    # Preserve original score ordering
    return sorted(seen.values(), key=lambda h: h["score"])


class VectorStore:
    """
    Persistent ChromaDB collection wrapper.

    Parameters
    ----------
    db_path:         Directory for ChromaDB persistence. Pass None or ":memory:"
                     to use an ephemeral (in-memory) client — useful for tests.
    collection_name: Name of the ChromaDB collection.
    embedder:        OllamaEmbedder instance used for all embed operations.
    batch_size:      Number of documents to add per ChromaDB call.
    """

    def __init__(
        self,
        db_path: str | Path | None,
        collection_name: str,
        embedder: OllamaEmbedder,
        batch_size: int = 50,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder
        self.batch_size = batch_size

        if db_path is None or str(db_path) == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _get_or_create_collection(self) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedder,
            metadata={"created_at": datetime.now().isoformat()},
        )

    def reset(self) -> None:
        """Drop the existing collection and recreate it empty."""
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info("Deleted existing collection '%s'.", self.collection_name)
        except Exception:
            pass
        self._collection = self._get_or_create_collection()
        logger.info("Recreated collection '%s'.", self.collection_name)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add(self, documents: list[dict[str, Any]]) -> None:
        """
        Add a list of prepared documents to the collection in batches.

        Each document must have keys: id (str), text (str), metadata (dict).
        Embeddings are computed via the embedder and stored explicitly so
        that ChromaDB does not attempt to re-embed documents at query time.
        """
        total = len(documents)
        inserted = 0

        for start in range(0, total, self.batch_size):
            batch = documents[start : start + self.batch_size]
            ids = [d["id"] for d in batch]
            texts = [d["text"] for d in batch]
            metas = [d["metadata"] for d in batch]
            embeddings = self.embedder.embed(texts)

            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metas,
                embeddings=embeddings,
            )
            inserted += len(batch)
            logger.debug("Inserted %d / %d documents.", inserted, total)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        where: dict | None = None,
        deduplicate: bool = True,
        score_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic similarity search.

        Parameters
        ----------
        text:             Query string (embedded with the same model used at ingest).
        top_k:            Number of unique parent issues to return.
        where:            Optional ChromaDB metadata filter.
        deduplicate:      If True (default), keep only the best-scoring chunk per
                          parent issue_id so the same issue never dominates results.
                          When False, all matched chunks are returned as-is.
        score_threshold:  Maximum L2 distance to accept.  Results whose best score
                          exceeds this threshold are discarded and an empty list is
                          returned.  None (default) disables the threshold check.
                          Lower L2 distance = more similar; typical range 0–2.

        Returns
        -------
        List of result dicts, each with keys:
            id, text, metadata, score (lower = more similar for L2 distance).
        Returns an empty list if no results meet the score_threshold.
        """
        # Over-fetch so deduplication still leaves top_k unique parents.
        fetch_k = top_k * 4 if deduplicate else top_k
        vector = self.embedder.embed_one(text)
        kwargs: dict[str, Any] = {"query_embeddings": [vector], "n_results": fetch_k}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        hits = [
            {"id": id_, "text": doc, "metadata": meta, "score": dist}
            for id_, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

        if deduplicate:
            hits = _deduplicate_by_parent(hits)

        hits = hits[:top_k]

        # Apply score threshold: if the best result is still too distant,
        # the query has no relevant matches in the collection.
        if score_threshold is not None and hits:
            best_score = hits[0]["score"]  # sorted ascending after dedup
            if best_score > score_threshold:
                logger.info(
                    "Score threshold %.3f exceeded (best score %.3f); returning no results.",
                    score_threshold,
                    best_score,
                )
                return []

        return hits
