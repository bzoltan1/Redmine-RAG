# Redmine RAG

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system
built on top of a Redmine issue tracker. Designed for QA engineers at
SUSE/openSUSE who need to query historical issue knowledge in natural language
without cloud APIs, subscription costs, or data leaving the machine.

**Blog post:** https://bzoltan1.github.io/redmine-rag-system/

---

## How it works

1. **Download** — Fetch issues (with full journal/comment history) from
   multiple Redmine projects via the REST API.
2. **Anonymize** — Replace user names with `User_XXXXX` placeholders and scrub
   emails, IP addresses, internal hostnames, and openQA test-run URLs from
   free text.
3. **Ingest** — Split each issue into independently-embeddable chunks, embed
   with `nomic-embed-text` (via Ollama), and store in a local ChromaDB
   vector database.
4. **Query** — Answer natural-language questions by extracting metadata filters
   (status, priority, project) with an LLM, doing semantic retrieval from
   ChromaDB, and generating a grounded answer with `llama3` — all locally.

---

## Stack

| Component | Choice |
|---|---|
| Language | Python 3.13 |
| Vector store | ChromaDB (persistent, local) |
| Embedding model | `nomic-embed-text` via Ollama |
| Chat model | `llama3` via Ollama |
| Redmine source | progress.opensuse.org REST API |

No cloud APIs. No LangChain. No Docker required.

---

## Setup

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com) running locally
- A Redmine API key

### Install

```bash
git clone https://github.com/bzoltan1/Redmine-RAG
cd Redmine-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Pull Ollama models

```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### Configure

```bash
cp .env.example .env
# Edit .env — set REDMINE_API_KEY and PROJECT_IDS at minimum
```

Key `.env` variables:

| Variable | Description | Default |
|---|---|---|
| `REDMINE_BASE_URL` | Redmine instance URL | `https://progress.opensuse.org` |
| `REDMINE_API_KEY` | Your Redmine API key | (required) |
| `PROJECT_IDS` | Comma-separated project IDs to download | (required) |
| `EMBED_MODEL` | Ollama embedding model | `nomic-embed-text` |
| `CHAT_MODEL` | Ollama chat model | `llama3` |
| `TOP_K` | Number of issues to retrieve per query | `5` |
| `SCORE_THRESHOLD` | Max L2 distance to accept; empty disables the check | (disabled) |
| `JOURNALS_ONLY_PROJECTS` | Projects where only issues with comments are kept | `openqatests` |
| `DEV_PROJECT_ID` | Project used in `--dev` mode | `qesecurity` |

---

## Pipeline

Run the four pipeline steps in order. Each step is resumable if interrupted.

### Step 1 — Download

```bash
# Full download (first time)
python pipeline/01_download.py

# Run in background (logs to logs/download.log)
python pipeline/01_download.py --background

# Incremental sync after first run
python pipeline/01_download.py --sync

# First incremental sync from a specific date
python pipeline/01_download.py --since 2025-12-01
```

### Step 2 — Anonymize

```bash
python pipeline/02_anonymize.py
```

Replaces user names and scrubs PII from all free-text fields.
Produces `data/redmine_anonymized.json` and `data/user_mapping.json`.

### Step 3 — Ingest

```bash
python pipeline/03_ingest.py

# Reset and re-ingest from scratch (needed after dataset size changes)
python pipeline/03_ingest.py --reset
```

Chunks issues (description + journal batches), embeds with `nomic-embed-text`,
and upserts into ChromaDB. Fully resumable without `--reset`.

### Step 4 — Query

```bash
# Interactive REPL
python pipeline/04_query.py

# Single question
python pipeline/04_query.py --query "What kernel issues are open?"

# Batch mode from a file (one question per line)
python pipeline/04_query.py --queries-file my_questions.txt

# Skip LLM filter extraction (~5s saved on CPU-only hardware)
python pipeline/04_query.py --no-filter

# Show retrieved source issues alongside the answer
python pipeline/04_query.py --show-sources
```

All pipeline scripts accept `--dev` to use an isolated dev environment
(`data/dev/`, single small project) without touching production data.

---

## Development mode

```bash
# Download dev data (qesecurity project only)
python pipeline/01_download.py --dev

# Anonymize dev data
python pipeline/02_anonymize.py --dev

# Ingest dev data
python pipeline/03_ingest.py --dev

# Query dev collection
python pipeline/04_query.py --dev
```

---

## Evaluation

Run retrieval evaluation against a set of benchmark questions:

```bash
# Evaluate using tests/eval/questions.jsonl
python tests/eval/hit_rate.py

# With custom top-K and no filter extraction
python tests/eval/hit_rate.py --top-k 10 --no-filter
```

Edit `tests/eval/questions.jsonl` to add questions with known expected issue
IDs (`expected_ids`). Questions without expected IDs are still run but marked
as N/A for hit-rate calculation.

---

## Tests

```bash
# Run all unit + integration tests
pytest tests/

# With coverage
pytest tests/ --cov=core --cov-report=term-missing
```

Current: **279 tests, 97% coverage** on `core/`.

---

## Architecture

```
app.py            Flask web interface — form, answer, source cards, /status, /eta
templates/
  index.html      Jinja2 template — vanilla JS + CSS, no external frameworks

core/
  anonymizer.py   User field anonymization + PII scrubbing + openQA URL stripping
  checkpoint.py   Atomic download checkpoints + incremental sync state
  document.py     Issue → chunks (description chunk + journal chunks)
  embedder.py     OllamaEmbedder wrapping nomic-embed-text
  rag.py          Full RAG loop: filter extraction (status/priority/project) → retrieve → generate
  redmine_client.py  Redmine REST API client with adaptive exponential backoff
  store.py        ChromaDB wrapper (upsert, deduplication, score threshold)
  timing.py       StageTimer, ProgressBar, PipelineReport

pipeline/
  01_download.py  Full + incremental download, crash-safe, --background
  02_anonymize.py Two-pass anonymization + PII scrubbing
  03_ingest.py    Chunk + embed + upsert into ChromaDB
  04_query.py     Interactive REPL, single query, batch mode (--queries-file)

tests/
  unit/           One test file per core module (279 tests, 97% coverage)
  integration/    End-to-end anonymize → ingest → query pipeline test
  eval/           Retrieval evaluation framework (hit_rate.py + questions.jsonl)
```

---

## Performance (CPU-only hardware)

| Stage | Throughput | Notes |
|---|---|---|
| Download | ~0.5–1 issues/s | Network-bound at 0.5s rate limit |
| Anonymize | ~1,400 issues/s | CPU-bound, trivially fast |
| Embed + ingest | ~3.2 chunks/s | `nomic-embed-text` on CPU (~6h for 67k chunks) |
| Filter extraction | ~5s | One `llama3` chat call |
| Retrieval | ~0.2s | ChromaDB vector search |
| Answer generation | ~25–55s | `llama3` on CPU |

Use `--no-filter` to skip the 5s filter extraction step for faster queries.

---

## Web interface

A Flask web frontend is included, mirroring the style of the companion
Bugzilla RAG project.

```bash
# Start the web server (production collection, port 5000)
python app.py

# Dev collection
python app.py --dev

# Skip filter extraction (faster on CPU)
python app.py --no-filter

# Custom port
python app.py --port 8080
```

Then open `http://localhost:5000/` in a browser.

Features:
- Query form with placeholder examples
- Applied filters shown as badges (status / priority / project)
- Answer rendered as formatted HTML with issue citations
- Source issue cards with direct links to `progress.opensuse.org`
- Live status bar (idle / in-progress / estimated wait)
- "No results" warning when score threshold or empty retrieval occurs

Routes: `GET /` (form), `POST /` (query + answer), `GET /status`, `GET /eta`

---

## Monitoring background processes

```bash
# Download
tail -f logs/download.log

# Ingest
tail -f logs/ingest.log | grep -v HTTP
```
