#!/usr/bin/env python3
"""
pipeline/01_download.py — Download Redmine issues with journals.

Modes
-----
Full download (default)
    Downloads every issue for every configured project from scratch.
    Resumes automatically if interrupted.

Incremental sync (--since DATE or --sync)
    Fetches only issues updated on or after DATE, merges them into the
    existing master dataset, and updates the sync state.

    --since 2025-12-02          fetch issues updated >= that date
    --sync                      use the date recorded by the last --since/--sync run

    Use --sync for regular scheduled runs. Use --since for the first incremental
    run after migrating an existing dataset.

Flags
-----
--dev       Use dev-mode config (qesecurity project, data/dev/).
--since     ISO-8601 date for incremental sync (e.g. 2025-12-02).
--sync      Shorthand for --since <last_synced_at> from sync state file.

Timing summary is printed at the end.
Configuration comes from .env via config.py.
"""

import argparse
import json
import sys
import time
import logging
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def merge_into(existing: dict[int, dict], incoming: list[dict]) -> tuple[int, int]:
    """
    Merge *incoming* issues into the *existing* id→issue map in place.

    Returns (n_new, n_updated) counts.
    """
    n_new = n_updated = 0
    for issue in incoming:
        iid = issue["id"]
        if iid in existing:
            n_updated += 1
        else:
            n_new += 1
        existing[iid] = issue
    return n_new, n_updated


def load_master(c: PipelineConfig) -> dict[int, dict]:
    """Load the master file into an id→issue map."""
    if c.MASTER_FILE.exists():
        with c.MASTER_FILE.open("r", encoding="utf-8") as fh:
            return {i["id"]: i for i in json.load(fh)}
    return {}


def save_master(c: PipelineConfig, issues_map: dict[int, dict]) -> None:
    c.MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with c.MASTER_FILE.open("w", encoding="utf-8") as fh:
        json.dump(list(issues_map.values()), fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Download modes
# ---------------------------------------------------------------------------

def download_project_full(
    client: RedmineClient, c: PipelineConfig, project_id: str
) -> tuple[list[dict], float]:
    """Full download of all issues for *project_id*. Returns (issues, elapsed)."""
    t_start = time.perf_counter()
    cp_path = checkpoint_file(c, project_id)
    cp = ckpt.load(cp_path)

    total: int | None = cp.get("total")
    offset: int = cp.get("offset", 0)
    successful_ids: set[int] = set(cp.get("successful_ids", []))
    issues_map: dict[int, dict] = {
        issue["id"]: issue for issue in load_project_data(c, project_id)
    }

    if offset > 0 and offset == total:
        progress(f"  Bulk '{project_id}': already complete ({offset}/{total}), skipping.")
    elif offset > 0:
        progress(f"  Resuming '{project_id}' from offset {offset}/{total} ...")

    # ---- Paginated bulk download ----
    t_bulk = time.perf_counter()
    try:
        while total is None or offset < total:
            progress(f"  [{project_id}] Bulk: fetching issues {offset+1}–{min(offset+100, total or offset+100)} ...")
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
            ckpt.save(cp_path, {
                "offset": offset, "total": total,
                "successful_ids": list(successful_ids), "failed_ids": {},
            })
            save_project_data(c, project_id, list(issues_map.values()))

            elapsed = time.perf_counter() - t_bulk
            rate = offset / elapsed if elapsed > 0 else 0
            eta = (total - offset) / rate if rate > 0 else 0
            progress(
                f"  [{project_id}] Bulk: {offset}/{total} ({100*offset/total:.0f}%)  "
                f"{rate:.0f} issues/s  ETA {format_duration(eta)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)

    except Exception as exc:
        progress(f"  ERROR during bulk download for '{project_id}': {exc}")

    # ---- Per-issue re-fetch for missing journals ----
    missing = [i for i in issues_map.values() if not i.get("journals")]
    if missing:
        n = len(missing)
        progress(
            f"\n  [{project_id}] Journal re-fetch: {n} issues need journals "
            f"(ETA ~{format_duration(n * c.RATE_LIMIT_SECONDS)} at {c.RATE_LIMIT_SECONDS}s/issue)"
        )
        t_j = time.perf_counter()
        network_errors = 0
        for i, issue in enumerate(missing, 1):
            try:
                enriched = client.fetch_issue(issue["id"])
                if enriched is not ISSUE_NOT_FOUND and isinstance(enriched, dict):
                    enriched["project_identifier"] = project_id
                    issues_map[enriched["id"]] = enriched
                network_errors = 0  # reset on success
            except Exception as exc:
                network_errors += 1
                progress(
                    f"  [{project_id}] Network error on issue #{issue['id']}: {exc}"
                    f"  (skipping, will retry next run)"
                )
                save_project_data(c, project_id, list(issues_map.values()))
                if network_errors >= 3:
                    progress(
                        f"  [{project_id}] {network_errors} consecutive errors — "
                        f"pausing 30s before continuing..."
                    )
                    time.sleep(30)
                    network_errors = 0
                continue
            if i % c.SAVE_INTERVAL == 0:
                save_project_data(c, project_id, list(issues_map.values()))
            elapsed_j = time.perf_counter() - t_j
            rate_j = i / elapsed_j if elapsed_j > 0 else 0
            eta_j = (n - i) / rate_j if rate_j > 0 else 0
            progress(
                f"  [{project_id}] Journals: {i}/{n} ({100*i/n:.0f}%)  "
                f"{rate_j:.1f} issues/s  ETA {format_duration(eta_j)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)

        with_journals = sum(1 for i in issues_map.values() if i.get("journals"))
        progress(
            f"  [{project_id}] Journal coverage: {with_journals}/{len(issues_map)} "
            f"({100*with_journals/len(issues_map):.0f}%)"
        )

    all_issues = list(issues_map.values())
    save_project_data(c, project_id, all_issues)
    ckpt.delete(cp_path)

    elapsed = time.perf_counter() - t_start
    progress(f"  [{project_id}] Done: {len(all_issues)} issues in {format_duration(elapsed)}")
    return all_issues, elapsed


def sync_project(
    client: RedmineClient, c: PipelineConfig, project_id: str, since: str
) -> tuple[int, int, float]:
    """
    Incremental sync: fetch issues updated >= *since* and merge into the
    existing per-project file.

    Returns (n_new, n_updated, elapsed_seconds).
    """
    t_start = time.perf_counter()
    issues_map: dict[int, dict] = {
        i["id"]: i for i in load_project_data(c, project_id)
    }
    existing_count = len(issues_map)

    offset = 0
    total: int | None = None
    all_incoming: list[dict] = []

    progress(f"  [{project_id}] Syncing issues updated >= {since} ...")

    try:
        while total is None or offset < total:
            page_issues, page_total = client.fetch_updated_since(
                project_id, since=since, offset=offset, limit=100
            )
            if total is None:
                total = page_total
                if total == 0:
                    progress(f"  [{project_id}] No updates since {since}.")
                    break
                progress(f"  [{project_id}] {total} issues to sync.")
            if not page_issues:
                break

            all_incoming.extend(page_issues)
            offset += len(page_issues)

            elapsed = time.perf_counter() - t_start
            rate = offset / elapsed if elapsed > 0 else 0
            eta = (total - offset) / rate if rate > 0 else 0
            progress(
                f"  [{project_id}] Fetched: {offset}/{total} ({100*offset/total:.0f}%)  "
                f"{rate:.0f} issues/s  ETA {format_duration(eta)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)

    except Exception as exc:
        progress(f"  ERROR syncing '{project_id}': {exc}")

    # ---- Per-issue re-fetch for missing journals in incoming issues ----
    # The bulk updated_on fetch includes journals, but some may be missing.
    missing = [i for i in all_incoming if not i.get("journals")]
    if missing:
        n = len(missing)
        progress(f"  [{project_id}] Journal re-fetch for {n} incoming issues ...")
        t_j = time.perf_counter()
        network_errors = 0
        for idx, issue in enumerate(missing, 1):
            try:
                enriched = client.fetch_issue(issue["id"])
                if enriched is not ISSUE_NOT_FOUND and isinstance(enriched, dict):
                    enriched["project_identifier"] = project_id
                    for i, item in enumerate(all_incoming):
                        if item["id"] == enriched["id"]:
                            all_incoming[i] = enriched
                            break
                network_errors = 0
            except Exception as exc:
                network_errors += 1
                progress(f"  [{project_id}] Network error on issue #{issue['id']}: {exc}  (skipping)")
                if network_errors >= 3:
                    progress(f"  [{project_id}] {network_errors} consecutive errors — pausing 30s...")
                    time.sleep(30)
                    network_errors = 0
                continue
            elapsed_j = time.perf_counter() - t_j
            rate_j = idx / elapsed_j if elapsed_j > 0 else 0
            eta_j = (n - idx) / rate_j if rate_j > 0 else 0
            progress(
                f"  [{project_id}] Journals: {idx}/{n} ({100*idx/n:.0f}%)  "
                f"{rate_j:.1f} issues/s  ETA {format_duration(eta_j)}"
            )
            time.sleep(c.RATE_LIMIT_SECONDS)

    n_new, n_updated = merge_into(issues_map, all_incoming)
    save_project_data(c, project_id, list(issues_map.values()))

    elapsed = time.perf_counter() - t_start
    progress(
        f"  [{project_id}] Sync done in {format_duration(elapsed)}: "
        f"+{n_new} new, {n_updated} updated  "
        f"(total {len(issues_map)} issues)"
    )
    return n_new, n_updated, elapsed


# ---------------------------------------------------------------------------
# Merge all per-project files into master
# ---------------------------------------------------------------------------

def merge_projects(c: PipelineConfig) -> list[dict]:
    master: list[dict] = []
    for pid in c.PROJECT_IDS:
        path = project_file(c, pid)
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                master.extend(json.load(fh))
    return master


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Redmine issues (full or incremental sync)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time full download
  python pipeline/01_download.py

  # First incremental sync after migrating an existing dataset
  python pipeline/01_download.py --since 2025-12-02

  # Regular scheduled sync (uses date from last run automatically)
  python pipeline/01_download.py --sync

  # Dev mode incremental sync
  python pipeline/01_download.py --dev --sync
""",
    )
    parser.add_argument(
        "--dev", action="store_true",
        help=f"Dev mode: use project '{cfg.DEV_PROJECT_ID}' and {cfg.DEV_DATA_DIR}.",
    )
    parser.add_argument(
        "--since", metavar="DATE",
        help=(
            "Incremental sync: fetch only issues updated on or after DATE "
            "(ISO-8601, e.g. 2025-12-02). Merges into existing data."
        ),
    )
    parser.add_argument(
        "--sync", action="store_true",
        help=(
            "Incremental sync using the date from the last successful sync. "
            "Equivalent to --since <last_synced_at>. "
            "Fails with an error if no previous sync state exists."
        ),
    )
    args = parser.parse_args()
    c = cfg.dev() if args.dev else cfg.prod()

    # Resolve sync date
    since: str | None = None
    if args.sync and args.since:
        parser.error("--sync and --since are mutually exclusive.")

    if args.sync:
        # Find the earliest last_synced_at across all projects (most conservative)
        states = [
            ckpt.get_last_synced_at(c.RAW_DIR, pid)
            for pid in c.PROJECT_IDS
        ]
        valid = [s for s in states if s]
        if not valid:
            parser.error(
                "--sync requires a previous sync state. "
                "Run with --since DATE first."
            )
        since = min(valid)  # earliest across projects = safest overlap point

    elif args.since:
        since = args.since

    mode = f"INCREMENTAL (since {since})" if since else "FULL DOWNLOAD"

    print(f"\n{SEPARATOR}")
    print(f"Redmine Issue Downloader  [{c.label()}]  [{mode}]")
    if c.is_dev:
        print(f"  Project  : {c.PROJECT_IDS}")
        print(f"  Data dir : {c.DATA_DIR}")
    if since:
        print(f"  Since    : {since}")
    print(f"{SEPARATOR}\n")

    if not c.REDMINE_API_KEY:
        progress("ERROR: REDMINE_API_KEY is not set. Check your .env file.")
        sys.exit(1)

    c.RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = RedmineClient(
        base_url=c.REDMINE_BASE_URL,
        api_key=c.REDMINE_API_KEY,
        rate_limit=c.RATE_LIMIT_SECONDS,
        timeout=c.REQUEST_TIMEOUT,
        max_retries=c.MAX_RETRIES,
    )

    report = PipelineReport(f"Download [{c.label()}] [{mode}]")
    total_new = total_updated = total_issues = 0

    # Record sync start time before any API calls (conservative: if we started
    # at T, next --sync will use T, so no gap can form between runs)
    sync_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if since:
        # ---- Incremental sync ----
        for pid in c.PROJECT_IDS:
            progress(f"\n--- Project: {pid} (sync) ---")
            n_new, n_updated, elapsed = sync_project(client, c, pid, since)
            total_new += n_new
            total_updated += n_updated
            report.record(f"Sync '{pid}'", elapsed)
            # Save sync state per project
            ckpt.save_sync_state(c.RAW_DIR, pid, sync_started_at)

        progress(
            f"\nSync complete: +{total_new} new, {total_updated} updated "
            f"across {len(c.PROJECT_IDS)} project(s)"
        )

    else:
        # ---- Full download ----
        for pid in c.PROJECT_IDS:
            progress(f"\n--- Project: {pid} ---")
            issues, elapsed = download_project_full(client, c, pid)
            total_issues += len(issues)
            report.record(f"Project '{pid}'", elapsed)
            ckpt.save_sync_state(c.RAW_DIR, pid, sync_started_at)

        progress(f"\nTotal issues downloaded: {total_issues}")

    # ---- Merge all per-project files into master ----
    progress("Merging all projects into master file ...")
    with StageTimer("Merge projects") as t_merge:
        master = merge_projects(c)
        save_master(c, {i["id"]: i for i in master})
    report.record("Merge projects", t_merge.elapsed)
    progress(f"Merged {len(master)} issues → {c.MASTER_FILE}")

    report.print()
    next_flag = " --dev" if c.is_dev else ""
    print(f"\nNext step: python pipeline/02_anonymize.py{next_flag}")
    print(f"Future syncs: python pipeline/01_download.py{next_flag} --sync")


if __name__ == "__main__":
    main()
