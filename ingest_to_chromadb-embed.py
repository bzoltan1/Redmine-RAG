#!/usr/bin/env python3
"""
Redmine -> ChromaDB ingestion WITH embeddings (Ollama Python SDK)

This script stores documents, metadata, AND embeddings into a ChromaDB collection.
Embeddings are computed at ingest time using Ollama Python SDK.

Requirements:
    pip install chromadb ollama

Usage:
    python3.11 ingest_to_chromadb_embed.py \
        --data redmine_master_dataset_anonymized.json \
        --db ./chroma_db \
        --collection redmine_issues \
        --batch 50 \
        --max-text 8192
"""

import json
import argparse
from datetime import datetime
from typing import Dict, List
import chromadb
from chromadb.config import Settings
import os
import sys

# -------------------------
# Embedding function
# -------------------------
from ollama import Client

class OllamaEmbeddingFunction:
    def __init__(self, model: str = "nomic-embed-text"):
        self.client = Client()
        self.model = model

    def __call__(self, input):
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
        resp = self.client.embed(model=self.model, input=inputs)
        return resp["embeddings"]  # check the actual key returned

# -------------------------
# Helpers
# -------------------------
def prepare_document(issue: Dict, max_text_len: int = 8192) -> Dict:
    """Convert a Redmine issue to a document text + metadata, truncate long text."""
    parts: List[str] = []

    subject = issue.get('subject', '') or ''
    description = issue.get('description', '') or ''

    parts.append(f"Subject: {subject}")
    if description:
        parts.append(f"Description: {description}")

    journals = issue.get('journals', []) or []
    if journals:
        parts.append("Comments:")
        for j in journals:
            notes = j.get('notes')
            if not notes:
                continue
            user = j.get('user', {}).get('name', 'Unknown')
            parts.append(f"- {user}: {notes}")

    document_text = "\n".join(parts).strip()
    if len(document_text) > max_text_len:
        document_text = document_text[:max_text_len] + "\n...[truncated]"

    metadata = {
        'issue_id': str(issue.get('id', '')),
        'subject': subject[:500],
        'status': issue.get('status', {}).get('name', ''),
        'priority': issue.get('priority', {}).get('name', ''),
        'tracker': issue.get('tracker', {}).get('name', ''),
        'project': issue.get('project', {}).get('name', ''),
        'project_id': issue.get('project_identifier', ''),
        'created_on': issue.get('created_on', ''),
        'updated_on': issue.get('updated_on', ''),
        'author': issue.get('author', {}).get('name', ''),
        'assigned_to': (issue.get('assigned_to') or {}).get('name', '') if issue.get('assigned_to') else '',
        'num_journals': len(journals),
    }

    doc_id = f"issue_{issue.get('id', '')}"

    return {
        'id': doc_id,
        'text': document_text,
        'metadata': metadata
    }

# -------------------------
# Main ingestion routine
# -------------------------
def ingest(args):
    data_file = args.data
    db_path = args.db
    collection_name = args.collection
    batch_size = args.batch
    max_text_len = args.max_text
    model_name = args.model

    print("\n" + "="*70)
    print("Redmine -> ChromaDB ingestion WITH embeddings (Ollama SDK)")
    print("="*70 + "\n")

    if not os.path.exists(data_file):
        print(f"ERROR: data file not found: {data_file}", file=sys.stderr)
        sys.exit(2)

    # Load data
    print(f"Loading {data_file}...")
    with open(data_file, "r", encoding="utf-8") as fh:
        try:
            issues = json.load(fh)
        except Exception as e:
            print("Failed to load JSON:", e, file=sys.stderr)
            sys.exit(2)
    total = len(issues)
    print(f"✓ Loaded {total} issues\n")

    # Initialize embedding function
    print(f"Initializing embedding model: {model_name} (Ollama SDK)")
    embedding_fn = OllamaEmbeddingFunction(model_name)
    print("✓ Embedding function ready\n")

    # Initialize ChromaDB
    print(f"Initializing ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        client.delete_collection(name=collection_name)
        print(f"⚠ Deleted existing collection '{collection_name}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"created_at": datetime.now().isoformat(), "ingested_with": "ollama-sdk"}
    )
    print(f"✓ Created collection '{collection_name}' with embeddings\n")

    # Batch add
    print(f"Processing issues (batch size: {batch_size})...\n")
    batch_ids: List[str] = []
    batch_docs: List[str] = []
    batch_metas: List[Dict] = []
    inserted = 0

    for idx, issue in enumerate(issues, start=1):
        doc = prepare_document(issue, max_text_len=max_text_len)

        batch_ids.append(doc['id'])
        batch_docs.append(doc['text'])
        batch_metas.append(doc['metadata'])

        if len(batch_ids) >= batch_size or idx == total:
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
            except Exception as e:
                print(f"ERROR during collection.add(): {e}", file=sys.stderr)

            inserted += len(batch_ids)
            print(f"  ✓ Progress: {inserted}/{total} ({100*inserted/total:.1f}%)")
            batch_ids, batch_docs, batch_metas = [], [], []

    print("\n" + "="*70)
    print("Ingestion complete with embeddings")
    print("="*70)
    try:
        count = collection.count()
        print(f"✓ Collection contains {count} documents")
    except Exception as e:
        print(f"Verification failed: {e}", file=sys.stderr)
    print("\n✓ ChromaDB ready for semantic search!\n")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest Redmine JSON into ChromaDB with embeddings (Ollama SDK)")
    parser.add_argument("--data", "-d", default="redmine_master_dataset_anonymized.json", help="Redmine JSON file")
    parser.add_argument("--db", default="./chroma_db", help="Chroma persistent directory")
    parser.add_argument("--collection", default="redmine_issues", help="Chroma collection name")
    parser.add_argument("--batch", type=int, default=50, help="Batch size")
    parser.add_argument("--max-text", type=int, default=8192, help="Max characters per document (truncation)")
    parser.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model")
    args = parser.parse_args()
    ingest(args)

if __name__ == "__main__":
    main()

