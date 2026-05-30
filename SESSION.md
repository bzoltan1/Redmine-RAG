# Redmine RAG — Session Memory

This file records the full context of the current session so work can be
resumed without loss of context.

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
| Embedding model | `nomic-embed-text` | via Ollama, ~3.2 chunks/s on this CPU |
| Chat model | `llama3` | CPU only, ~25–55s/query |
| Language | Python 3.13 | venv at `.venv/` |
| Test runner | pytest | 279 tests, 97% coverage |

---

## Directory structure

```
Redmine-RAG/
├── config.py                    # central config + dev/prod factory + SCORE_THRESHOLD
├── .env                         # API key, model names, paths (not committed)
├── .env.example                 # template
├── requirements.txt
├── README.md                    # full setup and usage documentation
├── PLAN.md                      # full design + implementation history
├── SESSION.md                   # this file
├── DEMO.md                      # demo script for QA engineers
├── logs/
│   ├── download.log             # download progress (--background or screen session)
│   └── ingest.log               # ingest progress
│
├── core/
│   ├── anonymizer.py            # user fields + PII regex + openQA URL stripping
│   ├── checkpoint.py            # download checkpoints + sync state
│   ├── document.py              # prepare() and prepare_chunks()
│   ├── embedder.py              # OllamaEmbedder, max_chars=1500, batch HTTP call
│   ├── rag.py                   # extract_filters (status/priority/project) + retrieve + generate
│   ├── redmine_client.py        # REST API + adaptive backoff + fetch_updated_since()
│   ├── store.py                 # ChromaDB wrapper, upsert, deduplication, score_threshold
│   └── timing.py                # StageTimer, ProgressBar, PipelineReport
│
├── app.py                       # Flask web interface (/, /status, /eta)
├── templates/
│   └── index.html               # Jinja2 template — form, answer, source cards
│
├── pipeline/
│   ├── 01_download.py           # full + incremental (--since/--sync) + --dev + --background
│   ├── 02_anonymize.py          # anonymize + PII scrub, --dev
│   ├── 03_ingest.py             # chunk + embed + upsert, --dev / --reset
│   └── 04_query.py              # RAG Q&A, --dev / --no-filter / --show-sources / --queries-file
│
├── tests/
│   ├── conftest.py
│   ├── unit/                    # 9 test files, one per core module
│   │   ├── test_anonymizer.py   # incl. openQA URL tests
│   │   ├── test_checkpoint.py
│   │   ├── test_document.py
│   │   ├── test_download.py
│   │   ├── test_embedder.py
│   │   ├── test_rag.py          # incl. project filter + score threshold tests
│   │   ├── test_redmine_client.py
│   │   ├── test_store.py        # incl. score threshold tests
│   │   └── test_timing.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── eval/
│       ├── questions.jsonl      # 10 seed benchmark questions
│       └── hit_rate.py          # retrieval evaluation script
│
└── data/                        # gitignored
    ├── raw/                     # per-project JSON + checkpoints + sync state
    ├── redmine_master.json      # merged prod dataset (29,544 issues)
    ├── redmine_anonymized.json  # anonymized (29,544 issues, 384 users)
    ├── user_mapping.json        # anonymization map
    ├── chroma_db/               # production ChromaDB (67,797 chunks — fully ingested)
    └── dev/                     # isolated dev data
        └── raw/qesecurity.json  # 765 issues
```

---

## Current data state (2026-05-30)

### Production download (`data/raw/`) — COMPLETE

| Project | Issues | Status |
|---|---|---|
| virtualization | 2,778 | done |
| performance | 2,076 | done |
| qesecurity | 765 | done |
| qe-kernel | 1,867 | done |
| qam | 561 | done |
| qe-yast | 2,431 | done |
| openqatests | 13,247 | done (journals-only filter: 7,593 kept) |
| openqa-infrastructure | ~1,242 | done (new — Phase 9 completion) |
| containers | 3,577 | done (new — Phase 9 completion) |

**Master dataset:** 29,544 issues (all 9 projects)
**Anonymized:** `data/redmine_anonymized.json` (29,544 issues, 384 users)
**ChromaDB:** 67,797 chunks, **fully ingested** (5h 52m at 3.2 chunks/s)

### Background processes

No background processes currently running. All pipeline steps complete.

### Resume commands (if re-ingest is needed)

```bash
# Re-anonymize (e.g. after Phase 10 openQA URL stripping — apply to fresh data)
.venv/bin/python pipeline/02_anonymize.py

# Re-ingest after anonymization change
screen -dmS redmine-ingest bash -c 'cd /home/balogh/Redmine-RAG && .venv/bin/python -u pipeline/03_ingest.py --reset 2>&1 | tee logs/ingest.log'

# Incremental sync
python pipeline/01_download.py --background --sync
```

---

## Key design decisions

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
Ingest is fully resumable at any point without `--reset`.

### Crash-resistant journal loop
Network errors in the journal re-fetch loop are caught per-issue.
Progress is saved, the issue is skipped, and the loop continues.
After 3 consecutive errors, a 30-second pause is inserted.

### openQA URL stripping
`openqa.suse.de` and `openqa.opensuse.org` URLs are stripped from all
free text before embedding (replaced with `[OPENQA-URL]`). Applied as
the first pass in `scrub_pii()` before the hostname regex, so no
hostname fragment from the URL leaks through.

### Score threshold
`store.query()` accepts `score_threshold: float | None`. When set, if the
best L2 distance exceeds the threshold the method returns an empty list.
Configure via `SCORE_THRESHOLD` in `.env` (unset = disabled).
Typical useful range for `nomic-embed-text`: `1.0`–`1.4`.

### Project filter extraction
`rag.extract_filters()` now extracts a third field `project` from natural
language. A `_PROJECT_ALIASES` dict (22 entries) maps aliases like
`"kernel"` → `"qe-kernel"`, `"containers"` → `"containers"`, etc.
Validated project names are applied as `project_id` ChromaDB filters.

### Stronger citation requirement
System prompt requires `(Issue #ID)` after every individual claim.
Prohibits uncited statements. Does not permit use of outside knowledge.

### Background download (`--background`)
`pipeline/01_download.py --background` re-execs the script detached
from the terminal using `subprocess.Popen(start_new_session=True)`,
appending stdout+stderr to `logs/download.log`. No `screen` needed.

---

## Performance characteristics (CPU-only hardware)

| Stage | Time | Notes |
|---|---|---|
| Journal re-fetch | ~0.5–1s/issue | Network-bound at 0.5s rate limit |
| Anonymize 29k issues | ~28s | CPU-bound, trivially fast |
| Embed + ingest 67,797 chunks | ~5h 52m | 3.2 chunks/s, CPU ceiling |
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

# Current result: 279 passed, 97% coverage
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
# SCORE_THRESHOLD=          # unset — disabled by default
```

---

## Completed work

### Phases 1–9
- [x] Full pipeline redesign (Phases 1–8)
- [x] 260 tests, 97% coverage (pre-Phase 10)
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
- [x] Full download of all 9 projects (29,544 issues total)
- [x] Full ingest of 67,797 chunks into ChromaDB (5h 52m)

### Phase 10 (completed 2026-05-30)
- [x] Strip openQA URLs from text before embedding (`anonymizer.py`)
- [x] Score threshold in `store.query()` (configurable via `SCORE_THRESHOLD`)
- [x] Project name extraction in `rag.extract_filters()` (22 aliases)
- [x] Stronger citation instruction in system prompt
- [x] `--queries-file` batch mode in `04_query.py`
- [x] `--background` flag in `01_download.py`
- [x] `tests/eval/` evaluation framework (`questions.jsonl` + `hit_rate.py`)
- [x] `README.md` — full documentation
- [x] 279 tests, 97% coverage (21 new tests added in Phase 10)

### Phase 11 (completed 2026-05-30)
- [x] `app.py` — Flask web interface (`/`, `/status`, `/eta`)
- [x] `templates/index.html` — Jinja2 template with form, answer, source cards
- [x] `Flask>=3.0` and `markdown2>=2.5` added to `requirements.txt`
- [x] Verified: starts cleanly, loads 67,797 docs, serves HTML (HTTP 200)
- [x] `SESSION.md`, `PLAN.md`, `README.md` updated

---

## Web interface (app.py)

Flask web frontend added in Phase 11. Mirrors the Bugzilla RAG project style.

```bash
python app.py                  # production, port 5000
python app.py --dev            # dev collection
python app.py --no-filter      # skip filter extraction
python app.py --port 8080
```

Open `http://localhost:5000/` in a browser.

Routes: `GET /` (form), `POST /` (query), `GET /status`, `GET /eta`

---

## Pending work (Phase 12 candidates)

- [ ] Re-run `02_anonymize.py` + `03_ingest.py --reset` to apply openQA URL
      stripping to the current dataset (anonymized before Phase 10 landed)
- [ ] Populate `tests/eval/questions.jsonl` with `expected_ids` by running
      queries and verifying results, to get meaningful hit rate @K metrics
- [ ] Tune `SCORE_THRESHOLD` using `hit_rate.py` to find the optimal value
- [ ] Streaming output via `ollama.chat(stream=True)` — pipe tokens to browser
      with Server-Sent Events so the answer appears word-by-word instead of
      after ~40s silence
- [ ] Conversation memory: add last-N-turns context for follow-up questions
- [ ] Production WSGI server: replace Flask dev server with `gunicorn` for
      multi-worker concurrent queries
