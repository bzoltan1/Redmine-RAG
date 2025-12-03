#!/usr/bin/env python3
import argparse
import chromadb
from chromadb.config import Settings
from datetime import datetime
from langchain_ollama import OllamaEmbeddings  # updated import

# -------------------------------------------------------------------
# Embedding wrapper for ChromaDB
# -------------------------------------------------------------------
class OllamaEmbeddingWrapper:
    def __init__(self, model: str):
        self.model = model
        self._embedder = OllamaEmbeddings(model=self.model)

    def name(self):
        return f"ollama-{self.model}"

    def __call__(self, input):
        if isinstance(input, str):
            return [self._embedder.embed_query(input)]
        return [self._embedder.embed_query(x) for x in input]

    def embed_documents(self, inputs):
        return [self._embedder.embed_query(doc) for doc in inputs]

    def embed_query(self, input):
        return [self._embedder.embed_query(input)]

# -------------------------------------------------------------------
# Querying utility
# -------------------------------------------------------------------
def run_query(collection, query_text, top_k):
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    ids = results["ids"][0]
    docs = results["documents"][0]
    dists = results["distances"][0]

    print("----------------------------------------------------------")
    print(f"Top {top_k} results:")
    print("----------------------------------------------------------")
    for i, (id_, doc, dist) in enumerate(zip(ids, docs, dists)):
        print(f"\n[{i+1}] ID = {id_}")
        print(f"Score = {dist:.4f}")
        print(f"Document:\n{doc}")
        print("----------------------------------------------------------")

# -------------------------------------------------------------------
# Interactive mode
# -------------------------------------------------------------------
def interactive_shell(client, collection_name, top_k):
    print("\n============================")
    print(" ChromaDB Query Interactive ")
    print("============================")
    print("Type 'exit' to quit\n")

    collection = client.get_collection(name=collection_name)

    while True:
        user_input = input("Query> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        run_query(collection, user_input, top_k)

# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Query ChromaDB with Ollama embeddings")
    parser.add_argument("--db", default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--collection", default="redmine_issues", help="Collection name")
    parser.add_argument("--model", default="mistral", help="Ollama embedding model (for new ingestion)")
    parser.add_argument("--topk", type=int, default=5, help="Number of results")
    parser.add_argument("--query", help="Run a single query and exit")
    args = parser.parse_args()

    # Connect to Chroma
    print(f"\nConnecting to ChromaDB at: {args.db}")
    client = chromadb.PersistentClient(path=args.db)

    # Run single query
    if args.query:
        collection = client.get_collection(name=args.collection)
        run_query(collection, args.query, args.topk)
    else:
        # Interactive mode
        interactive_shell(client, args.collection, args.topk)

if __name__ == "__main__":
    main()

