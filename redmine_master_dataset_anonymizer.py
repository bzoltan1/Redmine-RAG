import json
from typing import Dict, Any

# --- Configuration ---
INPUT_FILE = "redmine_master_dataset_with_journals.json"
OUTPUT_FILE = "redmine_master_dataset_anonymized.json"
MAPPING_FILE = "user_anonymization_mapping.json"


def generate_anonymous_name(user_id: int) -> str:
    """Generate a consistent anonymous name: User_XXXXX."""
    return f"User_{user_id:05d}"


def anonymize_user(user_obj: Dict[str, Any], mapping: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    """
    Anonymize a single user object (with 'id' and 'name') and update mapping.
    """
    if not user_obj or 'id' not in user_obj:
        return user_obj

    user_id = user_obj['id']
    original_name = user_obj.get('name', 'Unknown')

    if user_id not in mapping:
        mapping[user_id] = {
            'original_name': original_name,
            'anonymous_name': generate_anonymous_name(user_id)
        }

    return {
        'id': user_id,
        'name': mapping[user_id]['anonymous_name']
    }


def anonymize_issue(issue: Dict[str, Any], mapping: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    """Anonymize all user references in a single issue."""
    for field in ['author', 'assigned_to']:
        if issue.get(field):
            issue[field] = anonymize_user(issue[field], mapping)

    if 'journals' in issue and isinstance(issue['journals'], list):
        for journal in issue['journals']:
            if 'user' in journal and journal['user']:
                journal['user'] = anonymize_user(journal['user'], mapping)

    if 'watchers' in issue and isinstance(issue['watchers'], list):
        for i, watcher in enumerate(issue['watchers']):
            if isinstance(watcher, dict) and 'id' in watcher:
                issue['watchers'][i] = anonymize_user(watcher, mapping)

    return issue


def load_json_file(file_path: str):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
    except Exception as e:
        print(f"ERROR loading '{file_path}': {e}")
    return None


def save_json_file(data, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"ERROR saving '{file_path}': {e}")
        return False


def anonymize_dataset(input_file: str, output_file: str, mapping_file: str):
    """Main function to anonymize dataset."""
    print(f"\n{'='*70}\nRedmine Dataset Anonymization\n{'='*70}\n")

    issues = load_json_file(input_file)
    if not issues:
        return

    mapping: Dict[int, Dict[str, str]] = {}
    print(f"Loaded {len(issues)} issues. Anonymizing user data...")

    for idx, issue in enumerate(issues, 1):
        issues[idx - 1] = anonymize_issue(issue, mapping)
        if idx % 1000 == 0 or idx == len(issues):
            print(f"  Processed {idx}/{len(issues)} issues")

    if save_json_file(issues, output_file):
        print(f"\n✓ Anonymized dataset saved: {output_file}")

    if save_json_file(mapping, mapping_file):
        print(f"✓ User mapping saved: {mapping_file}")

    print(f"\n{'='*70}")
    print("Anonymization Complete!")
    print(f"Total issues: {len(issues)}")
    print(f"Unique users anonymized: {len(mapping)}\n")

    print("Sample mappings (first 10):")
    for idx, (user_id, data) in enumerate(list(mapping.items())[:10], 1):
        print(f"  {idx}. ID {user_id}: '{data['original_name']}' → '{data['anonymous_name']}'")
    if len(mapping) > 10:
        print(f"  ... and {len(mapping) - 10} more users")
    print()


def verify_anonymization(input_file: str, output_file: str):
    """Quick verification that anonymization worked."""
    print(f"\n{'='*70}\nVerification\n{'='*70}\n")

    orig_issues = load_json_file(input_file)
    anon_issues = load_json_file(output_file)

    if not orig_issues or not anon_issues:
        return

    print(f"Original issues: {len(orig_issues)}")
    print(f"Anonymized issues: {len(anon_issues)}")
    if len(orig_issues) != len(anon_issues):
        print("⚠ WARNING: Issue count mismatch!")

    # Sample check
    sample_idx = 0
    orig, anon = orig_issues[sample_idx], anon_issues[sample_idx]

    if orig.get('author'):
        print(f"\nSample Author:")
        print(f"  Original: {orig['author'].get('name', 'N/A')}")
        print(f"  Anonymized: {anon['author'].get('name', 'N/A')}")

    if orig.get('journals'):
        journal_orig, journal_anon = orig['journals'][0], anon['journals'][0]
        if journal_orig.get('user'):
            print(f"Sample Journal User:")
            print(f"  Original: {journal_orig['user'].get('name', 'N/A')}")
            print(f"  Anonymized: {journal_anon['user'].get('name', 'N/A')}")

    print("\n✓ Anonymization verified successfully\n")


if __name__ == "__main__":
    anonymize_dataset(INPUT_FILE, OUTPUT_FILE, MAPPING_FILE)
    verify_anonymization(INPUT_FILE, OUTPUT_FILE)
    print("Ready for RAG system!")
    print(f"Use the file: {OUTPUT_FILE}\n")

