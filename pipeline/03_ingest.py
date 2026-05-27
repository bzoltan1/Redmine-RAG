#!/usr/bin/env python3
"""
pipeline/03_ingest.py — Embed and ingest anonymized issues into ChromaDB.

Reads ANONYMIZED_FILE, splits each issue into section chunks, computes
embeddings in batches, and stores everything in ChromaDB.

Flags:
  --dev     Use dev-mode config (reads from data/dev/, uses collection
            'redmine_issues_dev'). Does not touch production data.
  --reset   Drop and recreate the collection before ingesting.

Timing summary and progress bar are shown throughout.
Configuration comes from .env via config.py.
"""

import argparse
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from config import PipelineConfig
from core.document import prepare_chunks
from core.embedder import OllamaEmbedder
from core.store import VectorStore
from core.timing import StageTimer, ProgressBar, PipelineReport

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 70


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Redmine issues into ChromaDB")
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            f"Dev mode: read from {cfg.DEV_DATA_DIR}, ingest into collection "
            f"'{cfg.DEV_COLLECTION_NAME}'. Does not touch production data."
        ),
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the ChromaDB collection before ingesting.",
    )
    args = parser.parse_args()
    c = cfg.dev() if args.dev else cfg.prod()

    print(f"\n{SEPARATOR}")
    print(f"Redmine -> ChromaDB Ingestor  [{c.label()}]")
    if c.is_dev:
        print(f"  Data dir   : {c.DATA_DIR}")
        print(f"  Collection : {c.COLLECTION_NAME}")
    print(f"{SEPARATOR}\n")

    if not c.ANONYMIZED_FILE.exists():
        log.error("Anonymized file not found: %s", c.ANONYMIZED_FILE)
        log.error(
            "Run: python pipeline/02_anonymize.py%s",
            "  --dev" if c.is_dev else "",
        )
        sys.exit(1)

    report = PipelineReport(f"Ingest [{c.label()}]")

    # --- Load ---
    with StageTimer("Load JSON") as t_load:
        with c.ANONYMIZED_FILE.open("r", encoding="utf-8") as fh:
            issues: list[dict] = json.load(fh)
    report.record("Load JSON", t_load.elapsed)
    log.info("  Loaded %d issues.", len(issues))

    # --- Init embedding model ---
    with StageTimer("Init embedder") as t_embed_init:
        embedder = OllamaEmbedder(model=c.EMBED_MODEL, host=c.OLLAMA_HOST)
    report.record("Init embedder", t_embed_init.elapsed)
    log.info("  Embedding model: %s", c.EMBED_MODEL)

    # --- Init ChromaDB ---
    with StageTimer("Init ChromaDB") as t_db_init:
        store = VectorStore(
            db_path=c.CHROMA_DIR,
            collection_name=c.COLLECTION_NAME,
            embedder=embedder,
            batch_size=c.BATCH_SIZE,
        )
        if args.reset:
            log.info("  --reset: dropping and recreating collection '%s'.", c.COLLECTION_NAME)
            store.reset()
    report.record("Init ChromaDB", t_db_init.elapsed)
    log.info("  Collection '%s': %d existing docs.", c.COLLECTION_NAME, store.count())

    # --- Chunking ---
    with StageTimer("Chunk issues") as t_chunk:
        all_chunks: list[dict] = []
        for issue in issues:
            all_chunks.extend(prepare_chunks(issue, max_text_len=c.MAX_TEXT_LEN))
    report.record("Chunk issues", t_chunk.elapsed)
    log.info("  Issues: %d → Chunks: %d", len(issues), len(all_chunks))

    # --- Embed + ingest with progress bar ---
    total = len(all_chunks)
    log.info("\n  Embedding and ingesting (batch size: %d)...", c.BATCH_SIZE)

    with ProgressBar(
        total=total,
        label="Embed+Ingest",
        unit="chunks",
        print_every=max(1, c.BATCH_SIZE),
    ) as bar:
        t_ingest = StageTimer("Embed + ingest")
        t_ingest.start()
        done = 0
        for start in range(0, total, c.BATCH_SIZE):
            batch = all_chunks[start: start + c.BATCH_SIZE]
            store.add(batch)
            done += len(batch)
            bar.update(done)

    report.record("Embed + ingest", t_ingest.elapsed)
    chunks_per_sec = total / t_ingest.elapsed if t_ingest.elapsed > 0 else 0

    # --- Summary ---
    print(f"\n{SEPARATOR}")
    print("Ingestion Complete!")
    print(f"{SEPARATOR}")
    print(f"  Collection     : {c.COLLECTION_NAME}")
    print(f"  Issues         : {len(issues)}")
    print(f"  Chunks stored  : {store.count()}")
    print(f"  Throughput     : {chunks_per_sec:.1f} chunks/s")
    print(f"  ChromaDB path  : {c.CHROMA_DIR}")

    report.print()
    print(f"\nNext step: python pipeline/04_query.py{'  --dev' if c.is_dev else ''}")


if __name__ == "__main__":
    main()
