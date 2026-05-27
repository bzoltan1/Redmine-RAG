#!/usr/bin/env python3
"""
pipeline/02_anonymize.py — Anonymize user fields in the master dataset.

Reads MASTER_FILE, runs two-pass anonymization (user fields + PII regex),
writes ANONYMIZED_FILE and USER_MAPPING_FILE.

Flags:
  --dev   Use dev-mode config (reads from data/dev/, writes to data/dev/).

Timing summary is printed at the end.
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
from core.anonymizer import anonymize_issue
from core.timing import StageTimer, ProgressBar, PipelineReport

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 70


def main() -> None:
    parser = argparse.ArgumentParser(description="Anonymize Redmine dataset")
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            f"Dev mode: read from {cfg.DEV_DATA_DIR}, write to {cfg.DEV_DATA_DIR}. "
            "Does not touch production data."
        ),
    )
    args = parser.parse_args()
    c = cfg.dev() if args.dev else cfg.prod()

    print(f"\n{SEPARATOR}")
    print(f"Redmine Dataset Anonymizer  [{c.label()}]")
    if c.is_dev:
        print(f"  Data dir : {c.DATA_DIR}")
    print(f"{SEPARATOR}\n")

    if not c.MASTER_FILE.exists():
        log.error("Master file not found: %s", c.MASTER_FILE)
        log.error(
            "Run: python pipeline/01_download.py%s",
            "  --dev" if c.is_dev else "",
        )
        sys.exit(1)

    report = PipelineReport(f"Anonymize [{c.label()}]")

    # --- Load ---
    with StageTimer("Load JSON") as t_load:
        with c.MASTER_FILE.open("r", encoding="utf-8") as fh:
            issues: list[dict] = json.load(fh)
    report.record("Load JSON", t_load.elapsed)
    log.info("  Loaded %d issues.", len(issues))

    # --- Anonymize ---
    mapping: dict[int, dict[str, str]] = {}
    anonymized: list[dict] = []

    with ProgressBar(
        total=len(issues),
        label="Anonymize",
        unit="issues",
        print_every=max(1, len(issues) // 100),
    ) as bar:
        t_anon = StageTimer("Anonymize issues")
        t_anon.start()
        for idx, issue in enumerate(issues, 1):
            anonymized.append(anonymize_issue(issue, mapping))
            bar.update(idx)

    report.record("Anonymize issues", t_anon.elapsed)

    # --- Save ---
    with StageTimer("Save JSON files") as t_save:
        c.ANONYMIZED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with c.ANONYMIZED_FILE.open("w", encoding="utf-8") as fh:
            json.dump(anonymized, fh, ensure_ascii=False, indent=2)
        with c.USER_MAPPING_FILE.open("w", encoding="utf-8") as fh:
            json.dump(mapping, fh, ensure_ascii=False, indent=2)
    report.record("Save JSON files", t_save.elapsed)

    # --- Summary ---
    print(f"\n{SEPARATOR}")
    print("Anonymization Complete!")
    print(f"{SEPARATOR}")
    print(f"  Issues processed       : {len(anonymized)}")
    print(f"  Unique users anonymized: {len(mapping)}")
    print(f"  Anonymized file        : {c.ANONYMIZED_FILE}")
    print(f"  User mapping file      : {c.USER_MAPPING_FILE}")

    if anonymized and issues:
        orig_author = (issues[0].get("author") or {}).get("name", "N/A")
        anon_author = (anonymized[0].get("author") or {}).get("name", "N/A")
        print(f"\n  Sample author:")
        print(f"    Original   : {orig_author}")
        print(f"    Anonymized : {anon_author}")

    report.print()
    print(f"\nNext step: python pipeline/03_ingest.py{'  --dev' if c.is_dev else ''}")


if __name__ == "__main__":
    main()
