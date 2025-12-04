from ollama import Client
import chromadb

ollama = Client()
chroma = chromadb.PersistentClient(path="./chroma_db")

collection = chroma.get_collection("redmine_issues")

query = "network interface fails after kernel update"

# Embed with the SAME model used during ingest
embedding = ollama.embeddings(
    model="nomic-embed-text",
    prompt=query
)["embedding"]

results = collection.query(
    query_embeddings=[embedding],
    n_results=5
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("-----")
    print(meta)
    print(doc[:200])

