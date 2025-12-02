#!/usr/bin/env python3
import requests
import json
import time
import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
REDMINE_BASE_URL = os.getenv("REDMINE_BASE_URL", "https://progress.opensuse.org")
PROJECT_IDS = os.getenv("PROJECT_IDS", "")
PROJECT_IDS = [pid.strip() for pid in PROJECT_IDS.split(",") if pid.strip()]
API_KEY = os.getenv("REDMINE_API_KEY")
LIMIT = 100
REQUEST_TIMEOUT = 30
MASTER_OUTPUT = "redmine_master_dataset.json"
CHECKPOINT_DIR = ".redmine_checkpoints"
ISSUES_ENDPOINT = f"{REDMINE_BASE_URL}/issues.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path(project_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{project_id}_checkpoint.json")


def load_json(path: str):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_complete(project_id: str) -> bool:
    """Check if project already has complete data."""
    data_file = f"redmine_{project_id}_issues_data.json"
    checkpoint = load_json(checkpoint_path(project_id))
    if os.path.exists(data_file):
        total_count = checkpoint.get("total") if checkpoint else None
        issues = load_json(data_file) or []
        if total_count is None or len(issues) == total_count:
            if os.path.exists(checkpoint_path(project_id)):
                os.remove(checkpoint_path(project_id))
            print(f"✓ Project '{project_id}' already complete. Skipping download.")
            return True
    return False


def fetch_issues(project_id: str) -> List[dict]:
    """Fetch all issues for a project with checkpoint/resume support."""
    if is_complete(project_id):
        return load_json(f"redmine_{project_id}_issues_data.json") or []

    checkpoint = load_json(checkpoint_path(project_id)) or {}
    offset = checkpoint.get("offset", 0)
    total = checkpoint.get("total", None)
    all_issues = load_json(f"redmine_{project_id}_issues_data.json") or []

    if offset > 0:
        print(f"\n⟳ Resuming download for project '{project_id}' from offset {offset}")

    headers = {'X-Redmine-API-Key': API_KEY, 'Content-Type': 'application/json'}

    while total is None or offset < total:
        params = {
            'project_id': project_id,
            'limit': LIMIT,
            'offset': offset,
            'include': 'journals',
            'status_id': '*',
            'sort': 'id:asc'
        }
        try:
            r = requests.get(ISSUES_ENDPOINT, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            issues = data.get('issues', [])
            for i in issues:
                i['project_identifier'] = project_id
            all_issues.extend(issues)

            if total is None:
                total = data.get('total_count', 0)
                if total == 0:
                    break

            offset += LIMIT
            save_json({"offset": offset, "total": total}, checkpoint_path(project_id))
            save_json(all_issues, f"redmine_{project_id}_issues_data.json")
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching '{project_id}' at offset {offset}: {e}")
            break

    if len(all_issues) == total:
        if os.path.exists(checkpoint_path(project_id)):
            os.remove(checkpoint_path(project_id))
        print(f"✓ Completed {len(all_issues)} issues for '{project_id}'")
    else:
        print(f"⚠ Partial fetch for '{project_id}': {len(all_issues)}/{total or '?'} issues")
    return all_issues


def merge_projects(project_ids: List[str], output_file: str):
    master = []
    for pid in project_ids:
        path = f"redmine_{pid}_issues_data.json"
        data = load_json(path) or []
        master.extend(data)
    save_json(master, output_file)
    print(f"✓ Merged {len(master)} issues into {output_file}")


if __name__ == "__main__":
    total_issues = 0
    for pid in PROJECT_IDS:
        issues = fetch_issues(pid)
        total_issues += len(issues)
    print(f"\nTotal issues downloaded: {total_issues}")
    merge_projects(PROJECT_IDS, MASTER_OUTPUT)
    print(f"Next step: Preprocess and chunk '{MASTER_OUTPUT}' for RAG.")

