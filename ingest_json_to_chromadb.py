"""
Redmine to ChromaDB Ingestion Script
=====================================
Converts Redmine JSON data into a ChromaDB vector database

Requirements:
pip install chromadb
"""

import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict
from datetime import datetime

# Configuration
REDMINE_DATA_FILE = "redmine_master_dataset_anonymized.json"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "redmine_issues"
BATCH_SIZE = 100


def prepare_document(issue: Dict) -> Dict:
    """Convert a Redmine issue into a document for ChromaDB"""
    
    # Build text content
    parts = []
    
    subject = issue.get('subject', '')
    description = issue.get('description', '')
    
    parts.append(f"Subject: {subject}")
    if description:
        parts.append(f"Description: {description}")
    
    # Add comments
    if 'journals' in issue and issue['journals']:
        parts.append("\nComments:")
        for journal in issue['journals']:
            if journal.get('notes'):
                user = journal.get('user', {}).get('name', 'Unknown')
                parts.append(f"{user}: {journal['notes']}")
    
    document_text = "\n".join(parts)
    
    # Build metadata
    metadata = {
        'issue_id': str(issue.get('id', '')),
        'subject': subject[:500] if subject else '',
        'status': issue.get('status', {}).get('name', ''),
        'priority': issue.get('priority', {}).get('name', ''),
        'tracker': issue.get('tracker', {}).get('name', ''),
        'project': issue.get('project', {}).get('name', ''),
        'project_id': issue.get('project_identifier', ''),
        'created_on': issue.get('created_on', ''),
        'updated_on': issue.get('updated_on', ''),
        'author': issue.get('author', {}).get('name', ''),
        'assigned_to': issue.get('assigned_to', {}).get('name', '') if issue.get('assigned_to') else '',
        'num_journals': len(issue.get('journals', [])),
    }
    
    doc_id = f"issue_{issue.get('id', '')}"
    
    return {
        'id': doc_id,
        'text': document_text,
        'metadata': metadata
    }


def ingest_to_chromadb():
    """Main ingestion function"""
    
    print(f"\n{'='*70}")
    print("Redmine to ChromaDB Ingestion")
    print(f"{'='*70}\n")
    
    # Load data
    print(f"Loading {REDMINE_DATA_FILE}...")
    with open(REDMINE_DATA_FILE, 'r', encoding='utf-8') as f:
        issues = json.load(f)
    print(f"✓ Loaded {len(issues)} issues\n")
    
    # Initialize ChromaDB
    print(f"Initializing ChromaDB...")
    print(f"  Directory: {CHROMA_PERSIST_DIR}")
    print(f"  Collection: {COLLECTION_NAME}\n")
    
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Delete existing collection if present
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"⚠ Deleted existing collection\n")
    except:
        pass
    
    # Create collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"created_at": datetime.now().isoformat()}
    )
    print(f"✓ Created collection\n")
    
    # Process and insert
    print(f"Processing issues (batch size: {BATCH_SIZE})...\n")
    
    total = len(issues)
    batch_ids = []
    batch_docs = []
    batch_metas = []
    inserted = 0
    
    for idx, issue in enumerate(issues, 1):
        doc = prepare_document(issue)
        
        batch_ids.append(doc['id'])
        batch_docs.append(doc['text'])
        batch_metas.append(doc['metadata'])
        
        # Insert when batch is full
        if len(batch_ids) >= BATCH_SIZE or idx == total:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            inserted += len(batch_ids)
            print(f"  ✓ Progress: {inserted}/{total} ({100*inserted/total:.1f}%)")
            
            batch_ids = []
            batch_docs = []
            batch_metas = []
    
    # Summary
    print(f"\n{'='*70}")
    print("Ingestion Complete!")
    print(f"{'='*70}")
    print(f"Documents inserted: {inserted}")
    print(f"Database location: {CHROMA_PERSIST_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    
    # Verify
    print(f"\nVerification:")
    count = collection.count()
    print(f"✓ Collection contains {count} documents")
    
    # Test query
    print(f"\nTesting query...")
    results = collection.query(
        query_texts=["performance issue"],
        n_results=3
    )
    print(f"✓ Query successful! Found {len(results['ids'][0])} results")
    
    if results['documents'][0]:
        print(f"\nSample result:")
        print(f"  {results['documents'][0][0][:150]}...")
    
    print(f"\n{'='*70}")
    print("✓ ChromaDB ready!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    ingest_to_chromadb()
