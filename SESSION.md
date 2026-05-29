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
| Embedding model | `nomic-embed-text` | via Ollama, ~2.8 chunks/s on this CPU |
| Chat model | `llama3` | CPU only, ~25–55s/query |
| Language | Python 3.13 | venv at `.venv/` |
| Test runner | pytest | 260 tests, 97% coverage |

---

## Directory structure

```
Redmine-RAG/
├── config.py                    # central config + dev/prod factory + JOURNALS_ONLY_PROJECTS
├── .env                         # API key, model names, paths (not committed)
├── .env.example                 # template
├── requirements.txt
├── PLAN.md                      # full design + implementation history
├── SESSION.md                   # this file
├── DEMO.md                      # demo script for QA engineers
├── logs/
│   ├── download.log             # live download progress (screen session)
│   └── ingest.log               # live ingest progress (screen session)
│
├── core/
│   ├── anonymizer.py            # user fields + PII regex + openQA URL stripping (TODO)
│   ├── checkpoint.py            # download checkpoints + sync state (save_sync_state etc.)
│   ├── document.py              # prepare() and prepare_chunks()
│   ├── embedder.py              # OllamaEmbedder, max_chars=1500, batch HTTP call
│   ├── rag.py                   # extract_filters + retrieve + generate + answer()
│   ├── redmine_client.py        # REST API + adaptive backoff + fetch_updated_since()
│   ├── store.py                 # ChromaDB wrapper, upsert, deduplication
│   └── timing.py                # StageTimer, ProgressBar, PipelineReport
│
├── pipeline/
│   ├── 01_download.py           # full + incremental (--since/--sync) + --dev + crash-safe
│   ├── 02_anonymize.py          # anonymize + PII scrub, --dev
│   ├── 03_ingest.py             # chunk + embed + upsert, --dev / --reset
│   └── 04_query.py              # RAG Q&A, --dev / --no-filter / --show-sources
│
├── tests/
│   ├── conftest.py
│   ├── unit/                    # 8 test files, one per core module
│   │   └── test_download.py     # merge_into() tests
│   └── integration/
│       └── test_pipeline.py
│
└── data/                        # gitignored
    ├── raw/                     # per-project JSON + checkpoints + sync state
    ├── redmine_master.json      # merged prod dataset (18,069 issues currently)
    ├── redmine_anonymized.json  # anonymized (18,069 issues)
    ├── chroma_db/               # production ChromaDB (13,700 chunks, growing)
    └── dev/                     # isolated dev data
        └── raw/qesecurity.json  # 765 issues, 98% journal coverage
```

---

## Current data state (2026-05-29)

### Production download (`data/raw/`)

| Project | Issues | Journals | Status |
|---|---|---|---|
| virtualization | 2,778 | ~74% done | journal re-fetch in progress |
| performance | 2,076 | 93% | done |
| qesecurity | 765 | 98% | done |
| qe-kernel | 1,867 | ~80% | journal re-fetch in progress |
| qam | 561 | 76% | done |
| qe-yast | 2,429 | 99% | done |
| openqatests | 13,247 | 57% (journals-only filter will keep ~7,600) | in progress |
| openqa-infrastructure | 0 | — | not started |
| containers | 0 | — | not started |

**Master dataset:** 18,069 issues merged from 7 projects
**Anonymized:** `data/redmine_anonymized.json` (18,069 issues, 301 users)
**ChromaDB:** 13,700 / 40,509 chunks embedded (~34%, ingest running)

### Background processes

Both running unattended in `screen` sessions:

```bash
screen -S redmine-download   # pipeline/01_download.py
screen -S redmine-ingest     # pipeline/03_ingest.py
```

Monitor:
```bash
tail -f logs/download.log
tail -f logs/ingest.log | grep -v HTTP
screen -ls
```

### Resume commands (if processes die)

```bash
# Resume download
screen -dmS redmine-download bash -c 'cd /home/balogh/Redmine-RAG && .venv/bin/python -u pipeline/01_download.py >> logs/download.log 2>&1'

# Resume ingest (no --reset needed — upsert is safe)
screen -dmS redmine-ingest bash -c 'cd /home/balogh/Redmine-RAG && .venv/bin/python -u pipeline/03_ingest.py 2>&1 | tee -a logs/ingest.log'
```

---

## Key design decisions made during this session

### openqatests filter (`JOURNALS_ONLY_PROJECTS`)
69% of openqatests issues are auto-generated failure tickets with no human
commentary. Decision: keep only issues with ≥1 journal entry.
Controlled by `JOURNALS_ONLY_PROJECTS=openqatests` in `.env`.

### Incremental sync (`--since` / `--sync`)
`updated_on >= DATE` captures both new issues and updated existing ones in
a single API call. Sync state stored in `data/raw/<project>_sync.json`.
Use `--since DATE` for the first sync after migrating existing data,
then `--sync` for all subsequent runs.

### Upsert instead of add
`collection.upsert()` replaces `collection.add()` in `store.py`.
Ingest is now fully resumable at any point without `--reset`.

### Crash-resistant journal loop
Network errors in the journal re-fetch loop are caught per-issue.
Progress is saved, the issue is skipped, and the loop continues.
After 3 consecutive errors, a 30-second pause is inserted.

### openQA URL stripping (TODO — not yet implemented)
Strip `openqa.suse.de` and `openqa.opensuse.org` URLs from text before
embedding. To be added to `core/anonymizer.py` as a new regex pattern.

### Score threshold (TODO — not yet implemented)
`store.query()` should return empty results + "no relevant issues found"
message when all top-K scores exceed a minimum L2 distance threshold.

### Project filter extraction (TODO — not yet implemented)
`rag.extract_filters()` should also extract project names from natural
language queries and apply as a `project_id` ChromaDB where= clause.

### Stronger citation instruction (TODO — not yet implemented)
System prompt should require `(Issue #ID)` after every claim.

### Batch query mode (TODO — not yet implemented)
`pipeline/04_query.py --queries-file questions.txt` for benchmarking.

### Evaluation framework (TODO — not yet implemented)
`tests/eval/questions.jsonl` with `{question, expected_issue_ids[]}`.
A hit-rate script runs queries and reports top-K recall.

### README (TODO — not yet written)
Full documentation: setup → pipeline steps → all flags → troubleshooting.

---

## Performance characteristics (CPU-only hardware)

| Stage | Time | Notes |
|---|---|---|
| Journal re-fetch | ~1s/issue | Network-bound at 0.5s rate limit |
| Anonymize 18k issues | ~25s | CPU-bound, trivially fast |
| Embed + ingest 40,509 chunks | ~4h | 2.8 chunks/s, CPU ceiling |
| Filter extraction | ~5s | One llama3 chat call |
| Retrieval | ~0.2s | ChromaDB vector search |
| Answer generation | ~25–55s | llama3 on CPU, output-length dependent |

**Demo tip:** use `--no-filter` to skip filter extraction and `--show-sources`
to show retrieved issues. This gives a 30s total response instead of 35s.
Warn the audience about CPU-only generation time upfront.

---

## Environment

```bash
# Activate venv
cd /home/balogh/Redmine-RAG
source .venv/bin/activate

# Run all tests
pytest tests/

# Current result: 260 passed, 97% coverage
```

### Ollama models available

```
nomic-embed-text:latest    0.3 GB   (embedding, EMBED_MODEL)
mxbai-embed-large:latest   0.7 GB   (embedding, 2.6x slower)
llama3:latest              4.7 GB   (chat, CHAT_MODEL)
qwen2.5:7b                 4.7 GB   (chat, slower on this CPU)
qwen3-coder:30b           18.6 GB   (too large for CPU-only)
qwen3-embedding:latest     4.7 GB   (not yet benchmarked)
```

### `.env` current values

```
REDMINE_BASE_URL=https://progress.opensuse.org
PROJECT_IDS=virtualization,performance,qesecurity,qe-kernel,qam,qe-yast,openqatests,openqa-infrastructure,containers
REDMINE_API_KEY=<redacted>
EMBED_MODEL=nomic-embed-text
CHAT_MODEL=llama3
RATE_LIMIT_SECONDS=0.5
DATA_DIR=./data
JOURNALS_ONLY_PROJECTS=openqatests
DEV_PROJECT_ID=qesecurity
DEV_DATA_DIR=./data/dev
DEV_COLLECTION_NAME=redmine_issues_dev
```

---

## Completed work

- [x] Full pipeline redesign (Phases 1–8)
- [x] 260 tests, 97% coverage
- [x] Adaptive backoff in Redmine client
- [x] PII scrubbing (emails, IPs, hostnames)
- [x] Section-based chunking (5 journals/chunk)
- [x] Chunk deduplication by parent issue
- [x] LLM-driven metadata filter extraction (status + priority)
- [x] Richer prompt context (dates, citation instruction)
- [x] Timing and progress indicators in all pipeline scripts
- [x] `--dev` mode with isolated data directory
- [x] `--no-filter` flag for faster CPU queries
- [x] Incremental sync (`--since` / `--sync`)
- [x] Crash-resistant journal re-fetch loop
- [x] `collection.upsert()` for resumable ingest
- [x] openqatests journals-only filter
- [x] First demo presented to QA engineers (success)
- [x] DEMO.md — demo script with 6 prepared queries and Q&A

## Pending work (from design interview)

- [ ] Strip openQA URLs from text before embedding
- [ ] Score threshold in `store.query()`
- [ ] Project name extraction in `rag.extract_filters()`
- [ ] Stronger citation instruction in system prompt
- [ ] `--queries-file` batch mode in `04_query.py`
- [ ] `--background` flag in `01_download.py`
- [ ] `tests/eval/` evaluation framework
- [ ] `README.md` — full documentation
- [ ] Complete download (openqa-infrastructure + containers still pending)
- [ ] Complete ingest (40,509 chunks, ~34% done)
