#!/usr/bin/env python3
"""
pipeline/04_query.py — Interactive and single-shot RAG Q&A interface.

Flags:
  --dev          Use dev-mode config (queries 'redmine_issues_dev' collection).
  --query / -q   Single question (non-interactive mode).
  --show-sources Print the retrieved source issues alongside the answer.
  --no-filter    Skip LLM-based metadata filter extraction (~5s saved on CPU).

Per-stage timing (filter extraction / retrieval / generation) is printed
after every answer.

Configuration comes from .env via config.py.
"""

import argparse
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from config import PipelineConfig
from core.embedder import OllamaEmbedder
from core.store import VectorStore
from core.rag import extract_filters, retrieve, build_prompt, generate
from core.timing import StageTimer, format_duration

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "-" * 70


def print_sources(retrieved: list[dict]) -> None:
    print(f"\n{SEPARATOR}")
    print(f"Sources ({len(retrieved)} issues retrieved):")
    print(SEPARATOR)
    for r in retrieved:
        meta = r.get("metadata") or {}
        print(
            f"  Issue #{meta.get('issue_id', '?')} | "
            f"{meta.get('project', '?')} | "
            f"{meta.get('status', '?')} | "
            f"score: {r.get('score', 0):.4f}"
        )
        print(f"  Subject: {meta.get('subject', '')[:80]}")
    print(SEPARATOR)


def print_timing(t_filter: float, t_retrieve: float, t_generate: float) -> None:
    total = t_filter + t_retrieve + t_generate
    print(f"\n{SEPARATOR}")
    print("Timing:")
    if t_filter > 0:
        print(f"  Filter extraction : {format_duration(t_filter)}")
    print(f"  Retrieval         : {format_duration(t_retrieve)}")
    print(f"  Generation        : {format_duration(t_generate)}")
    print(f"  Total             : {format_duration(total)}")
    print(SEPARATOR)


def run_query(
    c: PipelineConfig,
    store: VectorStore,
    question: str,
    show_sources: bool,
    use_filters: bool,
) -> None:
    print(f"\nQuestion: {question}")
    print(SEPARATOR)

    try:
        # Stage 1: filter extraction (optional)
        if use_filters:
            with StageTimer("Filter extraction") as t_f:
                filters = extract_filters(
                    question, model=c.CHAT_MODEL, host=c.OLLAMA_HOST
                )
        else:
            t_f = StageTimer("Filter extraction")
            t_f.elapsed = 0.0
            filters = {}

        # Build ChromaDB where= clause
        where = None
        if filters:
            if len(filters) == 1:
                key, val = next(iter(filters.items()))
                where = {key: {"$eq": val}}
            else:
                where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
            print(f"Filters applied: {filters}")

        # Stage 2: retrieval
        with StageTimer("Retrieval") as t_r:
            retrieved = retrieve(question, store, top_k=c.TOP_K, where=where)
            if not retrieved and where is not None:
                log.info("Filter produced no results; retrying without filter.")
                retrieved = retrieve(question, store, top_k=c.TOP_K)
                filters = {}

        # Stage 3: generation
        prompt = build_prompt(question, retrieved)
        with StageTimer("Generation") as t_g:
            answer_text = generate(
                prompt, model=c.CHAT_MODEL, host=c.OLLAMA_HOST
            )

    except Exception as exc:
        print(f"Error: {exc}")
        return

    print(f"\nAnswer:\n{answer_text}")

    if show_sources:
        print_sources(retrieved)

    print_timing(t_f.elapsed, t_r.elapsed, t_g.elapsed)


def interactive_loop(
    c: PipelineConfig,
    store: VectorStore,
    show_sources: bool,
    use_filters: bool,
) -> None:
    print("\n" + "=" * 70)
    print(f" Redmine RAG — Interactive Q&A  [{c.label()}]")
    print("=" * 70)
    print(f"  Embedding model : {c.EMBED_MODEL}")
    print(f"  Chat model      : {c.CHAT_MODEL}")
    print(f"  Collection      : {c.COLLECTION_NAME}  ({store.count()} docs)")
    print(f"  Top-K results   : {c.TOP_K}")
    print(f"  Filter extract  : {'enabled' if use_filters else 'disabled (--no-filter)'}")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            question = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        run_query(c, store, question, show_sources, use_filters)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Redmine RAG system")
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            f"Dev mode: query collection '{cfg.DEV_COLLECTION_NAME}' "
            f"in {cfg.DEV_DATA_DIR}. Does not touch production data."
        ),
    )
    parser.add_argument("--query", "-q", help="Single question (non-interactive mode)")
    parser.add_argument(
        "--show-sources", action="store_true",
        help="Print the retrieved source issues alongside the answer.",
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help=(
            "Skip LLM-based metadata filter extraction before retrieval. "
            "Saves ~5s per query on CPU-only hardware."
        ),
    )
    args = parser.parse_args()
    c = cfg.dev() if args.dev else cfg.prod()

    print(f"\n  [{c.label()}] collection: {c.COLLECTION_NAME}")

    with StageTimer("Init store") as t_init:
        embedder = OllamaEmbedder(model=c.EMBED_MODEL, host=c.OLLAMA_HOST)
        store = VectorStore(
            db_path=c.CHROMA_DIR,
            collection_name=c.COLLECTION_NAME,
            embedder=embedder,
        )

    print(f"  Store ready in {format_duration(t_init.elapsed)} ({store.count()} docs)")

    if store.count() == 0:
        print(
            f"\nWarning: collection '{c.COLLECTION_NAME}' is empty.\n"
            f"Run: python pipeline/03_ingest.py{'  --dev' if c.is_dev else ''}"
        )

    use_filters = not args.no_filter
    if not use_filters:
        print("  Filter extraction disabled (--no-filter).")

    if args.query:
        run_query(c, store, args.query, args.show_sources, use_filters)
    else:
        interactive_loop(c, store, args.show_sources, use_filters)


if __name__ == "__main__":
    main()
