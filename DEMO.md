# Redmine RAG — Demo Script for QA Engineers

## Setup (before the audience arrives)

```bash
cd /home/balogh/Redmine-RAG

# Verify the collection is ready
.venv/bin/python3 -c "
from core.embedder import OllamaEmbedder
from core.store import VectorStore
import config as cfg
c = cfg.prod()
store = VectorStore(db_path=c.CHROMA_DIR, collection_name=c.COLLECTION_NAME,
                    embedder=OllamaEmbedder(model=c.EMBED_MODEL, host=c.OLLAMA_HOST))
print(f'Ready: {store.count()} documents in collection')
"
```

Expected output: `Ready: ~40,000+ documents`

---

## What to say (2-minute intro)

> "This is a local RAG system built on top of our Redmine issue tracker.
> It has ingested 18,000 issues from 7 projects — virtualization, performance,
> qesecurity, qe-kernel, qam, qe-yast, and openqatests — including all their
> journal entries and comments. Everything runs locally: no cloud, no API costs,
> no data leaving this machine.
>
> You ask it a question in plain English. It finds the most relevant issues,
> passes them to a local LLM, and generates a grounded answer. Every claim it
> makes comes from our actual issue history."

---

## Demo queries (run in order — each shows a different capability)

### 1. Basic semantic search — finds what keyword search misses

```bash
.venv/bin/python pipeline/04_query.py \
  --query "What problems have been reported with kernel testing on s390x architecture?" \
  --show-sources --no-filter
```

**What to point out:** The system found issues even though the exact phrase
"s390x architecture" may not appear in every result — it understands semantic
similarity, not just keywords.

---

### 2. Automatic status filtering — no CLI flags needed

```bash
.venv/bin/python pipeline/04_query.py \
  --query "Are there any rejected issues related to network testing and why were they rejected?"
```

**What to point out:** Notice `Filters applied: {'status': 'Rejected'}` — the
system detected the intent from the question and pre-filtered the search before
retrieving. The LLM then explains the rejection reasons from the actual comments.

---

### 3. Cross-project pattern recognition

```bash
.venv/bin/python pipeline/04_query.py \
  --query "What security vulnerabilities or CVE-related issues have been tracked across projects?" \
  --show-sources --no-filter
```

**What to point out:** Results span qesecurity and other projects. The system
synthesises across project boundaries — something you cannot do in Redmine's
own search.

---

### 4. Engineering knowledge extraction from comments

```bash
.venv/bin/python pipeline/04_query.py \
  --query "How has the Agama installer testing been progressing and what issues were found?" \
  --show-sources --no-filter
```

**What to point out:** Agama is SLES 16's new installer. The answer is drawn
from the 276-comment epic in qe-yast. The LLM synthesises months of discussion
into a coherent summary with issue citations.

---

### 5. Practical operational question — shows real utility

```bash
.venv/bin/python pipeline/04_query.py \
  --query "What are the most common reasons for openQA test failures in virtualization?" \
  --show-sources --no-filter
```

**What to point out:** This is the kind of question a new QA engineer would
ask on their first week. The system gives a real, sourced answer in ~30 seconds
instead of requiring hours of Redmine archaeology.

---

### 6. Priority and urgency filtering

```bash
.venv/bin/python pipeline/04_query.py \
  --query "What high priority issues are currently open or in progress?"
```

**What to point out:** Shows `Filters applied: {'priority': 'High'}` or
`{'status': 'In Progress'}`. The system identified both the urgency and status
intent from the natural language.

---

## Likely questions and answers

**Q: Is the data up to date?**
> Currently ingested through late 2025 for 7 projects, with 3 more (openqa-infrastructure,
> containers, performance) downloading in the background. We have incremental
> sync built in — `python pipeline/01_download.py --sync` fetches only new and
> updated issues since the last run. Weekly syncs will take under 5 minutes.

**Q: What about data privacy?**
> All user names are replaced with anonymous identifiers (User_00123) before
> anything is embedded or stored. Email addresses, IP addresses, and internal
> hostnames in issue descriptions and comments are also redacted automatically.
> The anonymization mapping is kept locally and never committed to the repo.

**Q: Can it answer questions about specific issues?**
> Yes. Try: "Tell me about the samba_adcli failures on 15-SP6" — it will find
> the relevant issues and summarise what happened and how it was resolved.

**Q: Why does it take ~30 seconds to answer?**
> The LLM (llama3) runs entirely on CPU — no GPU on this machine. With a GPU,
> response time drops to 2–5 seconds. The retrieval itself (vector search) takes
> under 0.5 seconds — it's the answer generation that's slow.

**Q: Can I search for issues from a specific project?**
> Yes, just ask: "What qe-kernel issues are in progress?" — the system
> extracts the project name and filters accordingly.

**Q: How is this different from Redmine's built-in search?**
> Three key differences: (1) semantic understanding — finds issues by meaning,
> not just keyword match; (2) cross-project synthesis — answers span all projects
> at once; (3) generated answers — instead of a list of links, you get a
> coherent summary with citations.

---

## If something goes wrong

**LLM is slow / timing out:**
```bash
# Use --no-filter to skip the pre-retrieval LLM call (saves ~5s)
.venv/bin/python pipeline/04_query.py --query "..." --no-filter
```

**Collection is empty:**
```bash
# Check status
.venv/bin/python3 -c "
from core.store import VectorStore
from core.embedder import OllamaEmbedder
import config as cfg; c = cfg.prod()
s = VectorStore(c.CHROMA_DIR, c.COLLECTION_NAME, OllamaEmbedder(c.EMBED_MODEL, c.OLLAMA_HOST))
print(s.count(), 'documents')
" 2>/dev/null
```

**Ollama is not running:**
```bash
ollama serve &
# Wait 5 seconds then retry
```

---

## Key numbers to mention

| Metric | Value |
|---|---|
| Issues ingested | 18,069 |
| Projects covered | 7 (virtualization, performance, qesecurity, qe-kernel, qam, qe-yast, openqatests) |
| Document chunks in ChromaDB | ~40,500 |
| Journal entries indexed | ~131,000 |
| Retrieval time | < 0.5s |
| Answer generation | ~25–55s (CPU only) |
| Data leaves the machine | Never |
| Cloud API cost | €0 |
