#!/usr/bin/env python3
"""
pipeline/01_download.py — Download all Redmine issues with journals.

Flow for each configured project:
  1. Load checkpoint (resume if interrupted).
  2. Paginate through all issues with include=journals.
  3. For any issue returned without journals, re-fetch individually.
  4. Save checkpoint every SAVE_INTERVAL issues.
  5. Write <RAW_DIR>/<project_id>.json.
Merge all per-project files into MASTER_FILE when done.

Flags:
  --dev   Use dev-mode config (single small project, isolated data/dev/ dir).

Timing summary is printed at the end.
Configuration comes from .env via config.py.
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from config import PipelineConfig
from core.redmine_client import RedmineClient, ISSUE_NOT_FOUND
from core import checkpoint as ckpt
from core.timing import StageTimer, PipelineReport, format_duration

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEPARATOR = "=" * 70


def progress(msg: str) -> None:
    """Print a progress line immediately, bypassing any buffering."""
    print(msg, flush=True)


def project_file(c: PipelineConfig, project_id: str) -> Path:
    return c.RAW_DIR / f"{project_id}.json"


def checkpoint_file(c: PipelineConfig, project_id: str) -> Path:
    return c.RAW_DIR / f"{project_id}_checkpoint.json"


def load_project_data(c: PipelineConfig, project_id: str) -> list[dict]:
    path = project_file(c, project_id)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save_project_data(c: PipelineConfig, project_id: str, issues: list[dict]) -> None:
    path = project_file(c, project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(issues, fh, ensure_ascii=False, indent=2)


def download_project(
    client: RedmineClient, c: PipelineConfig, project_id: str
) -> tuple[list[dict], float]:
    """Download all issues for *project_id*. Returns (issues, elapsed_seconds)."""
    t_project_start = time.perf_counter()
    cp_path = checkpoint_file(c, project_id)
    cp = ckpt.load(cp_path)

    total: int | None = cp.get("total")
    offset: int = cp.get("offset", 0)
    successful_ids: set[int] = set(cp.get("successful_ids", []))
    issues_map: dict[int, dict] = {
        issue["id"]: issue for issue in load_project_data(c, project_id)
    }

    if offset > 0 and offset == total:
        progress(f"  Bulk download '{project_id}': already complete ({offset}/{total} issues), skipping.")
    elif offset > 0:
        progress(f"  Resuming '{project_id}' bulk download from issue {offset}/{total} ...")

    # ---- Paginated bulk download ----
    t_bulk_start = time.perf_counter()
    try:
        while total is None or offset < total:
            progress(f"  [{project_id}] Bulk: fetching issues {offset+1}–{offset+100} ...")
            page_issues, page_total = client.fetch_issues_page(
                project_id, offset=offset, limit=100
            )
            if total is None:
                total = page_total
                progress(f"  [{project_id}] Total issues: {total}")
            if not page_issues:
                break

            for issue in page_issues:
                issues_map[issue["id"]] = issue
                successful_ids.add(issue["id"])

            offset += len(page_issues)
            cp = {"offset": offset, "total": total,
                  "successful_ids": list(successful_ids), "failed_ids": {}}
            ckpt.save(cp_path, cp)
            save_project_data(c, project_id, list(issues_map.values()))

            elapsed = time.perf_counter() - t_bulk_start
            rate = offset / elapsed if elapsed > 0 else 0
            pct = 100 * offset / total if total else 0
            eta = (total - offset) / rate if rate > 0 else 0
            progress(
                f"  [{project_id}] Bulk: {offset}/{total} ({pct:.0f}%)  "
                f"{rate:.0f} issues/s  ETA {format_duration(eta)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)

    except Exception as exc:
        progress(f"  ERROR during bulk download for '{project_id}': {exc}")

    t_bulk = time.perf_counter() - t_bulk_start
    if t_bulk > 0.1:
        progress(f"  [{project_id}] Bulk download done in {format_duration(t_bulk)}")

    # ---- Per-issue re-fetch for missing journals ----
    missing_journals = [i for i in issues_map.values() if not i.get("journals")]
    t_journal = 0.0
    if missing_journals:
        n_missing = len(missing_journals)
        eta_estimate = format_duration(n_missing * c.RATE_LIMIT_SECONDS)
        progress(
            f"\n  [{project_id}] Journal re-fetch: {n_missing} issues need journals "
            f"(ETA ~{eta_estimate} at {c.RATE_LIMIT_SECONDS}s/issue)"
        )
        t_j_start = time.perf_counter()
        for i, issue in enumerate(missing_journals, 1):
            enriched = client.fetch_issue(issue["id"])
            if enriched is not ISSUE_NOT_FOUND and isinstance(enriched, dict):
                enriched["project_identifier"] = project_id
                issues_map[enriched["id"]] = enriched
            if i % c.SAVE_INTERVAL == 0:
                save_project_data(c, project_id, list(issues_map.values()))
            elapsed_j = time.perf_counter() - t_j_start
            rate_j = i / elapsed_j if elapsed_j > 0 else 0
            eta = (n_missing - i) / rate_j if rate_j > 0 else 0
            progress(
                f"  [{project_id}] Journals: {i}/{n_missing} ({100*i/n_missing:.0f}%)  "
                f"{rate_j:.1f} issues/s  ETA {format_duration(eta)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)
        t_journal = time.perf_counter() - t_j_start

        journals_found = sum(1 for i in issues_map.values() if i.get("journals"))
        progress(
            f"  [{project_id}] Journal coverage: {journals_found}/{len(issues_map)} "
            f"({100*journals_found/len(issues_map):.0f}%)"
        )

    all_issues = list(issues_map.values())
    save_project_data(c, project_id, all_issues)
    ckpt.delete(cp_path)

    t_project = time.perf_counter() - t_project_start
    log.info(
        "  Done: %d issues for '%s' in %s",
        len(all_issues), project_id, format_duration(t_project),
    )
    return all_issues, t_project


def merge_projects(c: PipelineConfig) -> list[dict]:
    master: list[dict] = []
    for pid in c.PROJECT_IDS:
        path = project_file(c, pid)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                master.extend(json.load(fh))
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Redmine issues")
    parser.add_argument(
        "--dev", action="store_true",
        help=(
            f"Dev mode: download only project '{cfg.DEV_PROJECT_ID}' "
            f"into {cfg.DEV_DATA_DIR}. Does not touch production data."
        ),
    )
    args = parser.parse_args()
    c = cfg.dev() if args.dev else cfg.prod()

    print(f"\n{SEPARATOR}")
    print(f"Redmine Issue Downloader  [{c.label()}]")
    if c.is_dev:
        print(f"  Project  : {c.PROJECT_IDS}")
        print(f"  Data dir : {c.DATA_DIR}")
    print(f"{SEPARATOR}\n")

    if not c.REDMINE_API_KEY:
        log.error("REDMINE_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    c.RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = RedmineClient(
        base_url=c.REDMINE_BASE_URL,
        api_key=c.REDMINE_API_KEY,
        rate_limit=c.RATE_LIMIT_SECONDS,
        timeout=c.REQUEST_TIMEOUT,
        max_retries=c.MAX_RETRIES,
    )

    report = PipelineReport(f"Download [{c.label()}]")
    total_issues = 0

    for pid in c.PROJECT_IDS:
        progress(f"\n--- Project: {pid} ---")
        issues, elapsed = download_project(client, c, pid)
        total_issues += len(issues)
        report.record(f"Project '{pid}'", elapsed)

    progress(f"\nTotal issues downloaded: {total_issues}")

    progress("Merging all projects into master file ...")
    with StageTimer("Merge projects") as t_merge:
        master = merge_projects(c)
        c.MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with c.MASTER_FILE.open("w", encoding="utf-8") as fh:
            json.dump(master, fh, ensure_ascii=False, indent=2)
    report.record("Merge projects", t_merge.elapsed)

    progress(f"Merged {len(master)} issues → {c.MASTER_FILE}")
    report.print()
    print(f"\nNext step: python pipeline/02_anonymize.py{'  --dev' if c.is_dev else ''}")


if __name__ == "__main__":
    main()
