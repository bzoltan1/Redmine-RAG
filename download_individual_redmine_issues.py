import requests
import json
import time
import os
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
REDMINE_BASE_URL = "https://progress.opensuse.org"
API_KEY = os.getenv("REDMINE_API_KEY")
REQUEST_TIMEOUT = 30

# Input and output files
MASTER_INPUT_FILE = "redmine_master_dataset.json"
ENRICHED_OUTPUT_FILE = "redmine_master_dataset_with_journals.json"
CHECKPOINT_FILE = "journal_enrichment_checkpoint.json"
PROGRESS_SAVE_INTERVAL = 50  # Save progress every N issues

# Rate limiting - be nice to the server
DELAY_BETWEEN_REQUESTS = 1.0  # seconds between each API call
DELAY_AFTER_ERROR = 10.0  # seconds to wait after an error
MAX_RETRIES_PER_ISSUE = 3  # Retry failed issues this many times


def load_master_dataset() -> List[Dict]:
    """Load the master dataset of issues."""
    try:
        with open(MASTER_INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"✓ Loaded {len(data)} issues from {MASTER_INPUT_FILE}")
            return data
    except FileNotFoundError:
        print(f"ERROR: File '{MASTER_INPUT_FILE}' not found!")
        return []
    except Exception as e:
        print(f"ERROR loading master dataset: {e}")
        return []


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint data if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                success_count = len(checkpoint.get('successfully_fetched', []))
                failed_count = len(checkpoint.get('failed_issues', {}))
                total = checkpoint.get('total_count', 0)
                print(f"✓ Loaded checkpoint: {success_count} successful, {failed_count} failed out of {total} total")
                return checkpoint
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None


def save_checkpoint(successfully_fetched: Set[int], failed_issues: Dict[int, int], total_count: int):
    """
    Save checkpoint data.
    - successfully_fetched: Set of issue IDs that were successfully enriched
    - failed_issues: Dict mapping issue_id -> retry_count for failed issues
    """
    checkpoint_data = {
        'successfully_fetched': list(successfully_fetched),
        'failed_issues': failed_issues,
        'total_count': total_count,
        'timestamp': time.time()
    }
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")


def save_enriched_data(issues: List[Dict], silent: bool = False):
    """Save enriched issues to output file."""
    try:
        with open(ENRICHED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(issues, f, ensure_ascii=False, indent=4)
        if not silent:
            print(f"✓ Enriched data saved to {ENRICHED_OUTPUT_FILE}")
    except Exception as e:
        print(f"ERROR saving enriched data: {e}")


def fetch_issue_with_journals(issue_id: int, api_key: str, retry_count: int = 0) -> Optional[Dict]:
    """
    Fetch a single issue with its journals included.
    Returns the issue data or None if failed.
    """
    url = f"{REDMINE_BASE_URL}/issues/{issue_id}.json"
    headers = {
        'X-Redmine-API-Key': api_key,
        'Content-Type': 'application/json'
    }
    params = {
        'include': 'journals,children,attachments,relations,changesets,watchers'
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        issue_data = data.get('issue', {})
        return issue_data
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"⚠ Issue #{issue_id} not found (404) - may have been deleted")
        elif response.status_code == 403:
            print(f"⚠ Issue #{issue_id} access forbidden (403)")
        elif response.status_code == 401:
            print(f"⚠ Authentication failed (401) - check API key")
        else:
            print(f"⚠ HTTP Error {response.status_code} for issue #{issue_id}: {e}")
        return None
        
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "Failed to resolve" in error_msg or "Name resolution" in error_msg:
            print(f"⚠ DNS resolution error for issue #{issue_id} (attempt {retry_count + 1}/{MAX_RETRIES_PER_ISSUE})")
        else:
            print(f"⚠ Request error for issue #{issue_id}: {e}")
        return None
        
    except Exception as e:
        print(f"⚠ Unexpected error for issue #{issue_id}: {e}")
        return None


def enrich_issues_with_journals(issues: List[Dict], api_key: str) -> List[Dict]:
    """
    Enrich all issues with journal data by fetching each issue individually.
    Supports resuming from checkpoint and retrying failed issues.
    """
    if not api_key:
        print("ERROR: API Key not found. Cannot proceed.")
        return issues

    total_count = len(issues)
    enriched_issues = []
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    if checkpoint:
        successfully_fetched = set(checkpoint.get('successfully_fetched', []))
        failed_issues = checkpoint.get('failed_issues', {})
        # Convert string keys back to integers
        failed_issues = {int(k): v for k, v in failed_issues.items()}
        print(f"\n⟳ Resuming from checkpoint")
        print(f"   - {len(successfully_fetched)} issues already successful")
        print(f"   - {len(failed_issues)} issues need retry")
    else:
        successfully_fetched = set()
        failed_issues = {}
    
    # Create a map for quick lookup
    issue_map = {issue['id']: issue for issue in issues}
    
    # Statistics
    issues_with_journals = 0
    issues_without_journals = 0
    issues_newly_failed = 0
    issues_retry_exhausted = 0
    
    print(f"\n{'='*70}")
    print(f"Starting Journal Enrichment Process")
    print(f"{'='*70}")
    print(f"Total issues to process: {total_count}")
    print(f"Already successful: {len(successfully_fetched)}")
    print(f"To retry: {len(failed_issues)}")
    print(f"Fresh attempts needed: {total_count - len(successfully_fetched) - len(failed_issues)}")
    print(f"Rate: 1 request every {DELAY_BETWEEN_REQUESTS} seconds")
    remaining = total_count - len(successfully_fetched)
    print(f"Estimated time: ~{(remaining * DELAY_BETWEEN_REQUESTS / 60):.1f} minutes")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for idx, issue in enumerate(issues, 1):
        issue_id = issue['id']
        
        # Skip if already successfully fetched
        if issue_id in successfully_fetched:
            # Load from enriched output if it exists, otherwise use original
            enriched_issues.append(issue)
            continue
        
        # Check if this issue has failed before and exceeded retry limit
        retry_count = failed_issues.get(issue_id, 0)
        if retry_count >= MAX_RETRIES_PER_ISSUE:
            print(f"[{idx}/{total_count}] Issue #{issue_id} - Max retries exceeded, keeping original")
            enriched_issues.append(issue)
            issues_retry_exhausted += 1
            continue
        
        # Attempt to fetch issue with journals
        status_prefix = f"[{idx}/{total_count}]"
        if retry_count > 0:
            print(f"{status_prefix} Retrying issue #{issue_id} (attempt {retry_count + 1}/{MAX_RETRIES_PER_ISSUE})...", end=" ")
        else:
            print(f"{status_prefix} Fetching issue #{issue_id}...", end=" ")
        
        enriched_issue = fetch_issue_with_journals(issue_id, api_key, retry_count)
        
        if enriched_issue:
            # Preserve the original project_identifier
            if 'project_identifier' in issue:
                enriched_issue['project_identifier'] = issue['project_identifier']
            
            journals = enriched_issue.get('journals', [])
            journal_count = len(journals)
            
            if journal_count > 0:
                print(f"✓ {journal_count} journal(s)")
                issues_with_journals += 1
            else:
                print(f"○ No journals")
                issues_without_journals += 1
            
            enriched_issues.append(enriched_issue)
            successfully_fetched.add(issue_id)
            
            # Remove from failed issues if it was there
            if issue_id in failed_issues:
                del failed_issues[issue_id]
        else:
            # Fetch failed - increment retry count
            print(f"✗ Failed")
            failed_issues[issue_id] = retry_count + 1
            enriched_issues.append(issue)  # Keep original
            issues_newly_failed += 1
        
        # Save checkpoint periodically
        processed_count = len(successfully_fetched) + len([fid for fid, cnt in failed_issues.items() if cnt >= MAX_RETRIES_PER_ISSUE])
        
        if idx % PROGRESS_SAVE_INTERVAL == 0:
            save_checkpoint(successfully_fetched, failed_issues, total_count)
            save_enriched_data(enriched_issues, silent=True)
            
            elapsed = time.time() - start_time
            completed = len(successfully_fetched)
            remaining_issues = total_count - completed
            
            print(f"\n--- Progress Checkpoint ---")
            print(f"Successful: {len(successfully_fetched)}/{total_count} ({100*len(successfully_fetched)/total_count:.1f}%)")
            print(f"With journals: {issues_with_journals} | Without: {issues_without_journals}")
            print(f"Currently failing: {len(failed_issues)} issues")
            print(f"Max retries exhausted: {issues_retry_exhausted} issues")
            print(f"Elapsed: {elapsed/60:.1f} min")
            print(f"---------------------------\n")
        
        # Rate limiting - be nice to the server
        if idx < total_count:  # Don't delay after the last issue
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Final save
    save_checkpoint(successfully_fetched, failed_issues, total_count)
    save_enriched_data(enriched_issues)
    
    # Check if we're done
    if len(failed_issues) == 0:
        # Clean up checkpoint file on successful completion
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print(f"✓ Checkpoint file removed (all issues processed successfully)")
    else:
        print(f"\n⚠ {len(failed_issues)} issues still failing. Run the script again to retry them.")
    
    # Final statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Journal Enrichment Run Complete!")
    print(f"{'='*70}")
    print(f"Successfully enriched: {len(successfully_fetched)}/{total_count} ({100*len(successfully_fetched)/total_count:.1f}%)")
    print(f"  - With journals: {issues_with_journals}")
    print(f"  - Without journals: {issues_without_journals}")
    print(f"Still failing: {len(failed_issues)} issues (will retry on next run)")
    print(f"Max retries exhausted: {issues_retry_exhausted} issues (kept original data)")
    print(f"Run time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}\n")
    
    return enriched_issues


def analyze_journal_coverage(issues: List[Dict]):
    """Analyze how many issues have journals and provide statistics."""
    print(f"\n{'='*70}")
    print("Journal Coverage Analysis")
    print(f"{'='*70}")
    
    total = len(issues)
    with_journals = 0
    total_journals = 0
    journal_distribution = {}
    
    for issue in issues:
        journals = issue.get('journals', [])
        journal_count = len(journals)
        total_journals += journal_count
        
        if journal_count > 0:
            with_journals += 1
            
        # Track distribution
        if journal_count not in journal_distribution:
            journal_distribution[journal_count] = 0
        journal_distribution[journal_count] += 1
    
    print(f"Total issues: {total}")
    print(f"Issues with journals: {with_journals} ({100*with_journals/total:.1f}%)")
    print(f"Issues without journals: {total - with_journals} ({100*(total-with_journals)/total:.1f}%)")
    print(f"Total journals fetched: {total_journals}")
    print(f"Average journals per issue: {total_journals/total:.2f}")
    
    print(f"\nJournal count distribution:")
    for count in sorted(journal_distribution.keys())[:10]:  # Show top 10
        num_issues = journal_distribution[count]
        print(f"  {count} journal(s): {num_issues} issues ({100*num_issues/total:.1f}%)")
    
    if len(journal_distribution) > 10:
        print(f"  ... and {len(journal_distribution) - 10} more categories")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Redmine Journal Enrichment Script (with Retry Logic)")
    print("="*70 + "\n")
    
    # Load master dataset
    issues = load_master_dataset()
    
    if not issues:
        print("No issues to process. Exiting.")
        exit(1)
    
    # Check if output file already exists
    if os.path.exists(ENRICHED_OUTPUT_FILE) and not os.path.exists(CHECKPOINT_FILE):
        response = input(f"\n'{ENRICHED_OUTPUT_FILE}' already exists and appears complete. Overwrite? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Exiting without changes.")
            exit(0)
    
    # Enrich issues with journals
    enriched_issues = enrich_issues_with_journals(issues, API_KEY)
    
    # Analyze results
    analyze_journal_coverage(enriched_issues)
    
    print(f"✓ Run complete! Enriched data saved to: {ENRICHED_OUTPUT_FILE}")
    
    # Check if there are failures that need retry
    checkpoint = load_checkpoint()
    if checkpoint and len(checkpoint.get('failed_issues', {})) > 0:
        print(f"\n💡 TIP: Run this script again to retry the {len(checkpoint['failed_issues'])} failed issues.")
    
    print()
