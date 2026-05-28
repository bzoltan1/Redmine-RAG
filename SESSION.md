# Redmine RAG — Session Memory

This file records the full context of the redesign and implementation session
so work can be resumed without loss of context.

---

## What this project is

A local RAG (Retrieval-Augmented Generation) system built on top of a Redmine
issue tracker database. It downloads issues from a Redmine instance, anonymizes
them, embeds them into ChromaDB using Ollama, and answers natural-language
questions by retrieving relevant issues and generating answers with a local LLM.

**Blog post:** https://bzoltan1.github.io/redmine-rag-system/
**Original repo:** https://github.com/bzoltan1/Redmine-RAG
**Working directory:** `/home/balogh/Redmine-RAG/`

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Vector store | ChromaDB (persistent) | `data/chroma_db/` |
| Embedding model | `nomic-embed-text` | via Ollama, ~47ms/call on this hardware |
| Chat model | `llama3` | CPU only, ~25–55s/query depending on output length |
| Language | Python 3.13 | venv at `.venv/` |
| Test runner | pytest | `238 tests, 97% coverage` |

---

## Directory structure

```
Redmine-RAG/
├── config.py                    # central config + dev/prod factory
├── .env                         # API key, model names, paths (not committed)
├── .env.example                 # template
├── requirements.txt
├── PLAN.md                      # full design + implementation history
├── SESSION.md                   # this file
│
├── core/
│   ├── anonymizer.py            # user fields + PII regex scrubbing
│   ├── checkpoint.py            # atomic checkpoint save/load/delete
│   ├── document.py              # prepare() and prepare_chunks()
│   ├── embedder.py              # OllamaEmbedder with batch + context cap
│   ├── rag.py                   # extract_filters + retrieve + generate + answer()
│   ├── redmine_client.py        # REST API client with adaptive backoff
│   ├── store.py                 # ChromaDB wrapper + deduplication
│   └── timing.py                # StageTimer, ProgressBar, PipelineReport
│
├── pipeline/
│   ├── 01_download.py           # download issues + journals, --dev flag
│   ├── 02_anonymize.py          # anonymize, --dev flag
│   ├── 03_ingest.py             # chunk + embed + ingest, --dev / --reset flags
│   └── 04_query.py              # RAG Q&A, --dev / --no-filter / --show-sources
│
├── tests/
│   ├── conftest.py              # shared fixtures (make_issue, make_journal, etc.)
│   ├── unit/                    # 7 test files, one per core module
│   └── integration/
│       └── test_pipeline.py     # end-to-end with in-memory ChromaDB
│
└── data/                        # gitignored
    ├── raw/                     # production per-project JSON + checkpoints
    ├── redmine_master.json      # merged prod dataset
    ├── redmine_anonymized.json  # anonymized prod dataset
    ├── chroma_db/               # production ChromaDB collection
    └── dev/                     # isolated dev data (separate from prod)
        ├── raw/
        │   ├── qesecurity.json            # 764 issues downloaded
        │   └── qesecurity_checkpoint.json # offset=764/764 bulk complete
        └── chroma_db/           # dev ChromaDB collection (not yet populated)
```

---

## Current data state

### Production (`data/`)
- **Project:** `qam` (561 issues)
- **Anonymized:** `data/redmine_anonymized.json` (561 issues, 91 users)
- **ChromaDB:** `data/chroma_db/`, collection `redmine_issues`, **1121 chunks**
- **Status:** fully operational, queryable

### Dev (`data/dev/`)
- **Project:** `qesecurity` (764 issues)
- **Bulk download:** complete (764/764)
- **Journal re-fetch:** partially complete — **634 issues have journals, 130 still pending**
- **Checkpoint:** `data/dev/raw/qesecurity_checkpoint.json` — offset=764, so bulk
  is recorded as done. The journal re-fetch tracks progress via the absence of
  `journals` in the saved JSON, not via checkpoint.
- **Anonymized:** not yet run
- **ChromaDB:** not yet ingested

### Resuming the dev pipeline

```bash
cd /home/balogh/Redmine-RAG

# Step 1: resume journal re-fetch (will pick up from 634/764)
# The script detects missing journals and re-fetches automatically
.venv/bin/python pipeline/01_download.py --dev

# Step 2: anonymize
.venv/bin/python pipeline/02_anonymize.py --dev

# Step 3: ingest
.venv/bin/python pipeline/03_ingest.py --dev --reset

# Step 4: query
.venv/bin/python pipeline/04_query.py --dev --query "Are there security CVEs tracked?"
.venv/bin/python pipeline/04_query.py --dev --no-filter --query "What issues are open?"
```

---

## Key design decisions made during the session

### Document chunking (`core/document.py`)
Each Redmine issue is split into multiple chunks:
- **Chunk 0** (`issue_<id>`): subject + description
- **Chunk N** (`issue_<id>_chunk_N`): subject + up to 5 journal entries

This ensures that content buried in comment #8 of a 20-comment issue is
independently retrievable. 561 issues → 1121 chunks.

### Deduplication (`core/store.py`)
`query()` over-fetches by 4× then keeps only the best-scoring chunk per parent
`issue_id`. Prevents one verbose issue from dominating all top-K results.

### LLM-driven metadata filters (`core/rag.py`)
Before retrieval, the question is sent to the chat LLM with a JSON extraction
prompt. Extracted `status` / `priority` values become ChromaDB `where=` clauses.
Handles "show rejected issues" without requiring explicit CLI flags.
Falls back to unfiltered search if filters produce no results.

### Adaptive backoff (`core/redmine_client.py`)
Base delay 0.5s. On HTTP 429/503: delay doubles (cap 60s). On success: resets
to 0.5s. Prevents IP bans while downloading as fast as the server allows.

### PII scrubbing (`core/anonymizer.py`)
Two passes: (1) replace user objects with `User_XXXXX`; (2) regex scrub of
`description` and journal `notes` for emails, IPv4 addresses, and internal
hostnames (`.suse.de`, `.suse.com`, `.internal`, `.local`).

### Embedder context cap (`core/embedder.py`)
Hard-truncate texts to 1500 chars before sending to Ollama. Prevents HTTP 400
from token-dense content (zypper progress bars, code blocks with repeated chars).
Texts are sent as a single batch call per ingest batch to minimise HTTP overhead.

### Dev / prod isolation (`config.py`)
`cfg.prod()` and `cfg.dev()` return `PipelineConfig` dataclass instances with
fully isolated paths and collection names. No `if dev` branching inside scripts.

---

## Performance characteristics (CPU-only hardware)

| Stage | Time | Notes |
|---|---|---|
| Journal re-fetch | ~2s/issue | Network-bound, rate-limited to 2s |
| Anonymize 561 issues | ~0.3s | CPU-bound, trivially fast |
| Embed + ingest 1121 chunks | ~5 min | ~47ms/call nomic-embed-text |
| Filter extraction | ~5s | One llama3 chat call |
| Retrieval | ~0.1s | ChromaDB vector search, negligible |
| Answer generation | ~25–55s | llama3 on CPU, output-length dependent |

`--no-filter` on step 4 skips the filter extraction call, saving ~5s per query
when status/priority filtering is not needed.

---

## Environment

```bash
# Activate venv
source /home/balogh/Redmine-RAG/.venv/bin/activate

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov-report=term-missing

# Current result: 238 passed, 97% coverage
```

### Ollama models available
```
nomic-embed-text:latest    0.3 GB   (embedding, default EMBED_MODEL)
mxbai-embed-large:latest   0.7 GB   (embedding, 2.6x slower, same context window)
llama3:latest              4.7 GB   (chat, default CHAT_MODEL)
qwen2.5:7b                 4.7 GB   (chat, slower than llama3 on this hardware)
qwen3-embedding:latest     4.7 GB   (embedding, not yet benchmarked)
qwen3-coder:30b           18.6 GB   (too large for CPU-only)
```

### `.env` current values
```
REDMINE_BASE_URL=https://progress.opensuse.org
PROJECT_IDS=qam
REDMINE_API_KEY=<redacted>
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=llama3
RATE_LIMIT_SECONDS=2.0
DATA_DIR=./data
DEV_PROJECT_ID=qesecurity
DEV_DATA_DIR=./data/dev
DEV_COLLECTION_NAME=redmine_issues_dev
```

---

## What works end-to-end (validated)

- [x] `pipeline/01_download.py` — bulk download + journal re-fetch + checkpoint resume
- [x] `pipeline/02_anonymize.py` — user fields + PII scrubbing
- [x] `pipeline/03_ingest.py` — chunking + embedding + ChromaDB ingest
- [x] `pipeline/04_query.py` — filter extraction + retrieval + LLM generation
- [x] `--dev` flag on all four scripts (isolation verified)
- [x] `--no-filter` flag on step 4
- [x] `--show-sources` flag on step 4
- [x] Adaptive backoff on 429/503 responses
- [x] Chunk deduplication by parent issue
- [x] Per-query timing breakdown

---

## openqatests project — data quality analysis

Run on 2026-05-27 against the 13,247 issues downloaded so far.

| Metric | openqatests | Other projects (avg) |
|---|---|---|
| Issues with journals | 20% | 90%+ |
| Auto-generated template (`## Observation`) | 69% | 8–27% |
| Journal text notes (vs field-change only) | 63% | 40–51% |
| Avg journal entries / issue | 1.5 | ~5–7 |
| Contains openQA URL links | 73% | rare |

**Finding:** 69% of openqatests issues are auto-generated from a standard
`## Observation / ## Reproducible / ## Expected result` template, populated
automatically when an openQA test fails. They contain a link to the failing
test run, a needle list, and a "fails since build X" statement. Most are
resolved silently — no human comments explaining the root cause or fix.

**The 20% with journals** contain genuine engineering discussion (the richest
issue had 200k chars of autoinst log dumps, 60+ comment threads on hard
boot failures, etc.). These are high signal. The 80% without journals are
mostly auto-created failure tickets resolved by fixing the needle or the
test, with no explanatory text.

**Decision pending:** whether to continue downloading openqatests journals
or exclude/filter the project. See interview section below.

---

## What was not done / future work

| Area | Notes |
|---|---|
| Run full 9-project pipeline | 28k issues already downloaded in repo root as `redmine_master_dataset_with_journals.json` (57 MB). Migrate to `data/` and run steps 2–3. |
| Finish qesecurity dev download | 130 issues still need journal re-fetch. Resume with `python pipeline/01_download.py --dev` |
| Conversation memory | Each query is stateless. Add last-N-turns to `messages[]` for follow-ups. |
| Streaming output | `ollama.chat(stream=True)` would show tokens as they arrive. |
| Web UI | FastAPI backend + minimal HTML frontend. |
| `qwen3-embedding` benchmark | Potentially better semantic quality; not yet tested. |
| README.md | Usage guide for new users has not been written. |

---

## Files created/modified in this session (relative to original repo)

**New files:**
- `config.py`
- `core/anonymizer.py`
- `core/checkpoint.py`
- `core/document.py`
- `core/embedder.py`
- `core/rag.py`
- `core/redmine_client.py`
- `core/store.py`
- `core/timing.py`
- `pipeline/01_download.py`
- `pipeline/02_anonymize.py`
- `pipeline/03_ingest.py`
- `pipeline/04_query.py`
- `tests/conftest.py`
- `tests/unit/test_anonymizer.py`
- `tests/unit/test_checkpoint.py`
- `tests/unit/test_document.py`
- `tests/unit/test_embedder.py`
- `tests/unit/test_rag.py`
- `tests/unit/test_redmine_client.py`
- `tests/unit/test_store.py`
- `tests/unit/test_timing.py`
- `tests/integration/test_pipeline.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`
- `PLAN.md`
- `SESSION.md` (this file)

**Deleted (replaced by new pipeline):**
- `download_redmine.py`
- `download_individual_redmine_issues.py`
- `redmine_master_dataset_anonymizer.py`
- `ingest_json_to_chromadb.py`
- `ingest_to_chromadb-embed.py`
- `query_cli.py`
- `test_chromadb_nomic.py`
