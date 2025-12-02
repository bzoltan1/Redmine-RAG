#!/usr/bin/env python3
import requests, json, time, os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

REDMINE_BASE_URL = os.getenv("REDMINE_BASE_URL","")
API_KEY = os.getenv("REDMINE_API_KEY")
MASTER_INPUT_FILE = "redmine_master_dataset.json"
ENRICHED_OUTPUT_FILE = "redmine_master_dataset_with_journals.json"
CHECKPOINT_FILE = "journal_enrichment_checkpoint.json"
REQUEST_TIMEOUT = 30
PROGRESS_SAVE_INTERVAL = 50
DELAY_BETWEEN_REQUESTS = 2.0
MAX_RETRIES = 3

def load_json(file: str):
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_json(data, file: str):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def fetch_issue(issue_id: int) -> Optional[Dict]:
    """Fetch single issue with journals and related info."""
    try:
        r = requests.get(
            f"{REDMINE_BASE_URL}/issues/{issue_id}.json",
            headers={'X-Redmine-API-Key': API_KEY, 'Content-Type': 'application/json'},
            params={'include': 'journals,children,attachments,relations,changesets,watchers'},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get('issue', None)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        msg = f"HTTP {code} for issue #{issue_id}" if code else str(e)
        print(f"⚠ {msg}")
    except Exception as e:
        print(f"⚠ Error fetching issue #{issue_id}: {e}")
    return None

def enrich_issues(issues: List[Dict]) -> List[Dict]:
    total = len(issues)
    checkpoint = load_json(CHECKPOINT_FILE) or {'status_map': {}, 'total': total}
    status_map = {int(k): v for k, v in checkpoint.get('status_map', {}).items()}
    enriched = []
    start_time = time.time()
    with_journals = 0
    newly_failed = 0
    max_retries_exceeded = 0

    print(f"\nStarting Journal Enrichment: {total} issues total")
    print(f"Resuming from checkpoint: {len(status_map)} issues already attempted\n")

    for idx, issue in enumerate(issues, 1):
        iid = issue['id']
        retries = status_map.get(iid, 0)

        if retries >= MAX_RETRIES:
            enriched.append(issue)
            max_retries_exceeded += 1
            print(f"[{idx}/{total}] Issue #{iid} - Max retries exceeded, keeping original")
            continue

        enriched_issue = fetch_issue(iid)
        if enriched_issue:
            if 'project_identifier' in issue:
                enriched_issue['project_identifier'] = issue['project_identifier']
            journal_count = len(enriched_issue.get('journals', []))
            with_journals += (journal_count > 0)
            enriched.append(enriched_issue)
            status_map[iid] = retries  # mark success
            print(f"[{idx}/{total}] Issue #{iid} ✓ Journals: {journal_count}")
        else:
            status_map[iid] = retries + 1
            enriched.append(issue)
            newly_failed += 1
            print(f"[{idx}/{total}] Issue #{iid} ✗ Failed (Retry {retries + 1}/{MAX_RETRIES})")

        # Save checkpoints periodically
        if idx % PROGRESS_SAVE_INTERVAL == 0 or idx == total:
            elapsed = (time.time() - start_time) / 60
            save_json({'status_map': status_map, 'total': total}, CHECKPOINT_FILE)
            save_json(enriched, ENRICHED_OUTPUT_FILE)
            print(f"--- Checkpoint saved at {idx}/{total} issues, elapsed ~{elapsed:.1f} min ---\n")

        if idx < total:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final save
    save_json({'status_map': status_map, 'total': total}, CHECKPOINT_FILE)
    save_json(enriched, ENRICHED_OUTPUT_FILE)

    if all(retries < MAX_RETRIES for retries in status_map.values()):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    elapsed_total = (time.time() - start_time) / 60
    print(f"\nEnrichment Complete: {len(enriched)}/{total} issues processed")
    print(f"With journals: {with_journals}, Failed: {newly_failed}, Max retries exceeded: {max_retries_exceeded}")
    print(f"Total runtime: {elapsed_total:.1f} minutes\n")
    return enriched

def analyze_journals(issues: List[Dict]):
    total = len(issues)
    with_journals = sum(1 for i in issues if i.get('journals'))
    total_journals = sum(len(i.get('journals', [])) for i in issues)
    print(f"Total issues: {total}")
    print(f"Issues with journals: {with_journals} ({100*with_journals/total:.1f}%)")
    print(f"Total journals fetched: {total_journals}, Avg per issue: {total_journals/total:.2f}")

if __name__ == "__main__":
    issues = load_json(MASTER_INPUT_FILE) or []
    if not issues:
        print("No issues found. Exiting."); exit(1)
    enriched = enrich_issues(issues)
    analyze_journals(enriched)

