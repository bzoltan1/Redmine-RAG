#!/usr/bin/env python3
"""
tests/eval/hit_rate.py — Retrieval evaluation script.

Runs each question from questions.jsonl through the RAG retrieval stage
and reports:
  - Per-question: whether expected issue IDs appear in the top-K results
  - Summary: hit rate @ K (proportion of questions with ≥1 expected ID found)

Usage
-----
  # Evaluate retrieval only (fast — no LLM generation call)
  python tests/eval/hit_rate.py

  # Evaluate with a custom top-K
  python tests/eval/hit_rate.py --top-k 10

  # Dev mode (uses dev collection)
  python tests/eval/hit_rate.py --dev

  # Disable metadata filter extraction
  python tests/eval/hit_rate.py --no-filter

  # Use a custom questions file
  python tests/eval/hit_rate.py --questions tests/eval/questions.jsonl

Output is written to tests/eval/results_<timestamp>.txt alongside stdout.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config as cfg
from core.embedder import OllamaEmbedder
from core.store import VectorStore
from core.rag import extract_filters, retrieve

SEPARATOR = "=" * 70


def load_questions(path: Path) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  WARNING: skipping malformed line: {e}")
    return questions


def evaluate_question(
    entry: dict,
    store: VectorStore,
    top_k: int,
    use_filters: bool,
    chat_model: str,
    ollama_host: str,
    score_threshold: float | None,
) -> dict:
    """
    Run retrieval for one question and return an evaluation result dict.
    """
    question = entry["question"]
    expected_ids = {str(i) for i in entry.get("expected_ids", [])}
    tags = entry.get("tags", [])

    t_start = time.perf_counter()

    # Extract filters (optional)
    filters: dict = {}
    if use_filters:
        try:
            filters = extract_filters(question, model=chat_model, host=ollama_host)
        except Exception:
            filters = {}

    # Build where= clause
    where = None
    if filters:
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            where = {key: {"$eq": val}}
        else:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    # Retrieve
    results = retrieve(question, store, top_k=top_k, where=where,
                       score_threshold=score_threshold)

    # Retry without filters if empty
    if not results and where is not None:
        results = retrieve(question, store, top_k=top_k, where=None,
                           score_threshold=score_threshold)
        filters = {}

    elapsed = time.perf_counter() - t_start

    retrieved_ids = {r["metadata"].get("issue_id", "") for r in results}
    best_score = results[0]["score"] if results else None

    # Determine hit: if expected_ids is empty, we can't evaluate — mark as N/A
    if not expected_ids:
        hit = None  # N/A — no ground truth provided
    else:
        hit = bool(expected_ids & retrieved_ids)

    return {
        "question": question,
        "tags": tags,
        "filters": filters,
        "retrieved_ids": sorted(retrieved_ids),
        "expected_ids": sorted(expected_ids),
        "best_score": best_score,
        "hit": hit,
        "elapsed": elapsed,
        "n_results": len(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval hit rate for the Redmine RAG system."
    )
    parser.add_argument("--dev", action="store_true", help="Use dev-mode config.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results to retrieve (default: 5).")
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Skip LLM-based filter extraction (faster).",
    )
    parser.add_argument(
        "--questions", default=str(Path(__file__).parent / "questions.jsonl"),
        metavar="FILE",
        help="Path to questions.jsonl (default: tests/eval/questions.jsonl).",
    )
    args = parser.parse_args()

    c = cfg.dev() if args.dev else cfg.prod()
    use_filters = not args.no_filter

    questions_path = Path(args.questions)
    if not questions_path.exists():
        print(f"ERROR: questions file not found: {questions_path}")
        sys.exit(1)

    questions = load_questions(questions_path)
    if not questions:
        print("No questions found.")
        sys.exit(0)

    print(f"\n{SEPARATOR}")
    print(f"Redmine RAG — Retrieval Evaluation  [{c.label()}]")
    print(f"{SEPARATOR}")
    print(f"  Questions file : {questions_path}")
    print(f"  Questions      : {len(questions)}")
    print(f"  Top-K          : {args.top_k}")
    print(f"  Filter extract : {'enabled' if use_filters else 'disabled'}")
    print(f"  Score threshold: {c.SCORE_THRESHOLD}")
    print(f"{SEPARATOR}\n")

    # Init store
    embedder = OllamaEmbedder(model=c.EMBED_MODEL, host=c.OLLAMA_HOST)
    store = VectorStore(
        db_path=c.CHROMA_DIR,
        collection_name=c.COLLECTION_NAME,
        embedder=embedder,
    )
    print(f"  Collection '{c.COLLECTION_NAME}': {store.count()} docs\n")

    results_list = []
    for i, entry in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {entry['question'][:80]}")
        result = evaluate_question(
            entry,
            store,
            top_k=args.top_k,
            use_filters=use_filters,
            chat_model=c.CHAT_MODEL,
            ollama_host=c.OLLAMA_HOST,
            score_threshold=c.SCORE_THRESHOLD,
        )
        results_list.append(result)

        # Per-question summary line
        hit_str = "HIT" if result["hit"] else ("MISS" if result["hit"] is False else "N/A")
        score_str = f"{result['best_score']:.4f}" if result["best_score"] is not None else "none"
        print(
            f"  {hit_str:4s}  results={result['n_results']}  "
            f"best_score={score_str}  filters={result['filters']}  "
            f"elapsed={result['elapsed']:.1f}s"
        )
        if result["retrieved_ids"]:
            print(f"  Retrieved IDs : {', '.join(result['retrieved_ids'][:10])}")
        if result["expected_ids"]:
            print(f"  Expected IDs  : {', '.join(result['expected_ids'])}")
        print()

    # Summary
    evaluable = [r for r in results_list if r["hit"] is not None]
    hits = [r for r in evaluable if r["hit"]]
    total_elapsed = sum(r["elapsed"] for r in results_list)

    print(f"\n{SEPARATOR}")
    print(f"  Evaluation Summary")
    print(f"{SEPARATOR}")
    print(f"  Total questions : {len(results_list)}")
    print(f"  Evaluable (with expected IDs): {len(evaluable)}")
    if evaluable:
        print(f"  Hit rate @{args.top_k}      : {len(hits)}/{len(evaluable)} ({100*len(hits)/len(evaluable):.0f}%)")
    print(f"  N/A (no ground truth)        : {len(results_list) - len(evaluable)}")
    print(f"  Total elapsed   : {total_elapsed:.1f}s")
    print(f"{SEPARATOR}")

    # Write results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"results_{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in results_list:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  Detailed results written to: {out_path}")


if __name__ == "__main__":
    main()
