#!/usr/bin/env python3
"""
app.py — Flask web interface for the Redmine RAG system.

Usage:
    python app.py [--port N] [--host ADDR] [--dev] [--no-filter] [--verbose]

Routes:
    GET  /        Query form (empty state)
    POST /        Run RAG query, render results
    GET  /status  JSON: {"processing": N}
    GET  /eta     JSON: {"eta": seconds}
"""

import argparse
import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import markdown2
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

import config as cfg
from core.embedder import OllamaEmbedder
from core.store import VectorStore
from core.rag import extract_filters, retrieve, build_prompt, generate

log = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state (process-scoped — single-worker Flask)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_processing = 0
_durations: deque = deque(maxlen=50)

_store: VectorStore | None = None
_config: cfg.PipelineConfig | None = None
_use_filters: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redmine_url(issue_id: str) -> str:
    base = _config.REDMINE_BASE_URL.rstrip("/") if _config else "https://progress.opensuse.org"
    return f"{base}/issues/{issue_id}"


def _format_sources(retrieved: list[dict]) -> list[dict]:
    """Convert raw VectorStore results to template-friendly dicts."""
    out = []
    for item in retrieved:
        meta = item.get("metadata") or {}
        issue_id = str(meta.get("issue_id", ""))
        out.append({
            "issue_id":   issue_id,
            "issue_url":  _redmine_url(issue_id),
            "subject":    str(meta.get("subject", ""))[:120],
            "project":    meta.get("project", ""),
            "project_id": meta.get("project_id", ""),
            "status":     meta.get("status", ""),
            "priority":   meta.get("priority", ""),
            "created_on": meta.get("created_on", "")[:10],
            "updated_on": meta.get("updated_on", "")[:10],
            "score":      f"{item.get('score', 0):.4f}",
            "snippet":    item.get("text", "").strip()[:600],
        })
    return out


def _run_query(question: str) -> dict:
    """
    Full RAG pipeline: filter extraction → retrieve → generate.
    Returns dict with answer, sources, elapsed, filters.
    """
    t0 = time.time()

    filters: dict = {}
    if _use_filters:
        filters = extract_filters(
            question,
            model=_config.CHAT_MODEL,
            host=_config.OLLAMA_HOST,
        )

    where = None
    if filters:
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            where = {key: {"$eq": val}}
        else:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    retrieved = retrieve(
        question, _store,
        top_k=_config.TOP_K,
        where=where,
        score_threshold=_config.SCORE_THRESHOLD,
    )

    if not retrieved and where is not None:
        log.info("Filter produced no results; retrying without filter.")
        retrieved = retrieve(
            question, _store,
            top_k=_config.TOP_K,
            score_threshold=_config.SCORE_THRESHOLD,
        )
        filters = {}

    prompt = build_prompt(question, retrieved)
    answer_text = generate(prompt, model=_config.CHAT_MODEL, host=_config.OLLAMA_HOST)
    answer_html = markdown2.markdown(answer_text)

    return {
        "answer":   answer_html,
        "sources":  _format_sources(retrieved),
        "elapsed":  f"{time.time() - t0:.1f}",
        "filters":  filters,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    global _processing

    question = answer = elapsed = error = ""
    sources: list = []
    filters: dict = {}

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            with _lock:
                _processing += 1
            t0 = time.time()
            try:
                result  = _run_query(question)
                answer  = result["answer"]
                sources = result["sources"]
                elapsed = result["elapsed"]
                filters = result["filters"]
            except Exception as exc:
                log.exception("Query failed")
                error = str(exc)
            finally:
                duration = time.time() - t0
                with _lock:
                    _processing -= 1
                    _durations.append(duration)
                if not elapsed:
                    elapsed = f"{duration:.1f}"

    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources,
        elapsed=elapsed,
        filters=filters,
        error=error,
        collection_size=_store.count() if _store else 0,
        top_k=_config.TOP_K if _config else 5,
        use_filters=_use_filters,
        mode=_config.label() if _config else "PROD",
    )


@app.route("/status")
def status():
    with _lock:
        return jsonify({"processing": _processing})


@app.route("/eta")
def eta():
    with _lock:
        avg = (sum(_durations) / len(_durations)) if _durations else 40.0
        estimate = round(_processing * avg, 1)
    return jsonify({"eta": estimate})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _store, _config, _use_filters

    parser = argparse.ArgumentParser(
        description="Redmine RAG web interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                        # production mode, port 5000
  python app.py --dev                  # dev collection (qesecurity)
  python app.py --no-filter --port 8080
""",
    )
    parser.add_argument("--dev", action="store_true",
                        help="Use dev-mode config (qesecurity collection).")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on (default: 5000).")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0).")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable LLM metadata filter extraction (~5s saved per query).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _config = cfg.dev() if args.dev else cfg.prod()
    _use_filters = not args.no_filter

    print(f"\n  Redmine RAG — Web Interface  [{_config.label()}]")
    print(f"  Collection : {_config.COLLECTION_NAME}")
    print(f"  ChromaDB   : {_config.CHROMA_DIR}")
    print(f"  Embed model: {_config.EMBED_MODEL}")
    print(f"  Chat model : {_config.CHAT_MODEL}")
    print(f"  Top-K      : {_config.TOP_K}")
    print(f"  Score thr. : {_config.SCORE_THRESHOLD}")
    print(f"  Filters    : {'enabled' if _use_filters else 'disabled (--no-filter)'}")

    print("\n  Initialising vector store...", end=" ", flush=True)
    embedder = OllamaEmbedder(model=_config.EMBED_MODEL, host=_config.OLLAMA_HOST)
    _store = VectorStore(
        db_path=_config.CHROMA_DIR,
        collection_name=_config.COLLECTION_NAME,
        embedder=embedder,
    )
    print(f"ready ({_store.count():,} docs)")

    if _store.count() == 0:
        print(
            f"\n  WARNING: collection '{_config.COLLECTION_NAME}' is empty.\n"
            f"  Run: python pipeline/03_ingest.py{'  --dev' if _config.is_dev else ''}"
        )

    print(f"\n  Serving at http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
