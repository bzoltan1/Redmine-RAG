# Redmine RAG System — Plan

## Overview

This document describes the design, implementation history, and current state
of the Redmine RAG system. The project started as a collection of one-off
Hackweek scripts and was redesigned into a clean, tested, maintainable pipeline
with a real RAG generation step: retrieve relevant issues from ChromaDB, then
generate a natural-language answer using a local Ollama LLM.

---

## Problems with the Original Design

| Problem | Detail |
|---|---|
| Split download logic | Two scripts did what should be one unified download + retry pass |
| Split ingest logic | Two ingestors differed only by a flag |
| Inconsistent embedding | Ingest used `ollama.Client`, query used `langchain_ollama`; incompatible |
| No generation step | Only semantic search, not RAG |
| No central config | `.env` in one script, hardcoded defaults in another |
| No data directory | All JSON files landed in repo root alongside source code |
| No tests | One ad-hoc smoke test, not a test suite |
| No `requirements.txt` | Dependencies undocumented |

---

## Design Goals

1. **Unified pipeline** — four numbered scripts that run in sequence
2. **Real RAG** — retrieve top-K issues, build a grounded prompt, generate an answer
3. **Central configuration** — one `config.py` reading from `.env`
4. **Testable core** — all business logic in `core/`; pipeline scripts are thin wrappers
5. **Proper test suite** — unit tests per module, integration tests for the full pipeline
6. **Clean data layout** — all data files under `data/`, gitignored
7. **Single dependency file** — `requirements.txt`
8. **Local-first** — ChromaDB + Ollama, no external API costs

---

## Directory Structure

```
Redmine-RAG/
│
├── PLAN.md                          # this file
├── requirements.txt                 # all Python dependencies
├── .env.example                     # template: copy to .env and fill in
├── .gitignore
│
├── config.py                        # central config; reads from .env
│
├── core/                            # business logic — testable, no side effects
│   ├── __init__.py
│   ├── redmine_client.py            # Redmine REST API client with adaptive backoff
│   ├── checkpoint.py                # atomic checkpoint load/save/delete
│   ├── anonymizer.py                # user field + free-text PII anonymization
│   ├── embedder.py                  # Ollama embedding wrapper (batch, context-safe)
│   ├── store.py                     # ChromaDB wrapper with chunk deduplication
│   ├── document.py                  # issue -> single doc or section chunks
│   └── rag.py                       # filter extraction + retrieve + generate
│
├── pipeline/                        # thin runner scripts; import from core/
│   ├── 01_download.py               # download all issues with journals + checkpoint/retry
│   ├── 02_anonymize.py              # anonymize user fields + PII scan in free text
│   ├── 03_ingest.py                 # chunk, embed, and ingest into ChromaDB
│   └── 04_query.py                  # RAG Q&A (single-shot or interactive REPL)
│
├── data/                            # gitignored; all data files land here
│   ├── raw/                         # per-project JSON + checkpoint files
│   ├── redmine_master.json          # merged dataset from all projects
│   ├── redmine_anonymized.json      # anonymized version of master
│   ├── user_mapping.json            # anonymization mapping (local only)
│   └── chroma_db/                   # ChromaDB persistent store
│
└── tests/
    ├── conftest.py                  # shared fixtures (synthetic issues, journals)
    ├── unit/
    │   ├── test_document.py         # prepare() and prepare_chunks() coverage
    │   ├── test_anonymizer.py       # user fields + PII regex scrubbing
    │   ├── test_checkpoint.py       # save/load/resume/delete
    │   ├── test_redmine_client.py   # pagination, retry, adaptive backoff
    │   ├── test_embedder.py         # batch embed, context cap, Ollama mocked
    │   ├── test_store.py            # add, query, deduplication (EphemeralClient)
    │   └── test_rag.py              # filter extraction, prompt, generate, answer()
    └── integration/
        └── test_pipeline.py         # end-to-end: anonymize → ingest → query → answer
```

---

## Configuration (`config.py`)

All tuneable values in one place, read from `.env` via `python-dotenv`.

| Variable | Default | Description |
|---|---|---|
| `REDMINE_BASE_URL` | `https://progress.opensuse.org` | Redmine instance URL |
| `REDMINE_API_KEY` | — | API key (never committed) |
| `PROJECT_IDS` | — | Comma-separated project slugs |
| `RATE_LIMIT_SECONDS` | `0.5` | Base inter-request delay (adaptive backoff floor) |
| `MAX_RETRIES` | `3` | Per-request retry attempts |
| `SAVE_INTERVAL` | `50` | Checkpoint every N issues |
| `DATA_DIR` | `./data` | Root for all data files |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3` | Ollama chat model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `BATCH_SIZE` | `50` | Ingest batch size |
| `MAX_TEXT_LEN` | `4000` | Document char limit (embedder applies 1500-char cap separately) |
| `TOP_K` | `5` | Unique parent issues returned per query |
| `COLLECTION_NAME` | `redmine_issues` | ChromaDB collection name |

---

## Module Responsibilities

### `core/redmine_client.py`

Wraps the Redmine JSON REST API. Handles pagination, per-issue fetching, and
**adaptive exponential backoff**:

- Base delay: `rate_limit` seconds (default 0.5s)
- On HTTP 429 or 503: delay doubles (capped at `backoff_max`, default 60s)
- On successful response: delay resets to `rate_limit`
- Network errors (DNS, timeout): fixed `retry_delay` between attempts

This allows downloading as fast as the server permits without risking an IP ban.

Key methods: `get_total_count`, `fetch_issues_page`, `fetch_issue`

---

### `core/checkpoint.py`

Atomic checkpoint persistence for the download stage. Writes to a `.tmp` sibling
file then renames, preventing corruption if the process is killed mid-write.

Key functions: `load`, `save`, `is_complete`, `delete`

---

### `core/anonymizer.py`

Two-pass anonymization — no file I/O:

**Pass 1 — structured user fields:** replaces `author`, `assigned_to`,
journal `user`, and `watchers` with `User_XXXXX` placeholders. The mapping
from real user ID to anonymous name is accumulated by the caller and can be
persisted to `user_mapping.json`.

**Pass 2 — free-text PII scan:** applies regex scrubbing to `description`
and journal `notes` fields, redacting:
- Email addresses → `[REDACTED-EMAIL]`
- IPv4 addresses → `[REDACTED-IP]`
- Internal hostnames (`.suse.de`, `.suse.com`, `.internal`, `.local`, etc.)
  → `[REDACTED-HOST]`

Matches are logged at DEBUG level for review. Content is redacted, not dropped.

Key functions: `anonymize_issue`, `anonymize_user`, `generate_anonymous_name`,
`scrub_pii`

---

### `core/document.py`

Converts Redmine issue dicts into ChromaDB documents. Two modes:

**`prepare(issue, max_text_len)`** — single document per issue. Subject +
description + all journals concatenated and hard-truncated. For short issues
or backward-compatible use.

**`prepare_chunks(issue, journals_per_chunk, max_text_len)`** — splits long
issues into multiple independently-embeddable chunks:

- **Chunk 0** (`issue_<id>`): subject + description
- **Chunk N** (`issue_<id>_chunk_N`): subject (for context) + N journal entries

All chunks carry the parent `issue_id` in their metadata so deduplication can
be applied at retrieval time. The pipeline uses `prepare_chunks` for ingestion.

---

### `core/embedder.py`

Single, consistent embedding implementation using `ollama.Client` directly.
Used at both ingest time and query time, eliminating any model mismatch.

**Context window safety:** texts are hard-truncated to `max_chars` (default
1500) before being sent to Ollama. This prevents HTTP 400 errors from
token-dense content (e.g. long code blocks, repeated progress-bar characters)
regardless of which embedding model is configured.

**Batch efficiency:** the `embed(texts)` method sends the full list in a single
`/api/embed` HTTP call rather than one call per text, eliminating per-call HTTP
overhead. Ollama processes inputs sequentially on the GPU regardless, so this
saves round-trip time without changing throughput.

**Embedding model choice:** `nomic-embed-text` is the default and recommended
model. On the tested hardware it embeds at ~47ms/call vs ~106ms/call for
`mxbai-embed-large`, making it ~2.3× faster with no measurable retrieval quality
difference for short-form issue text. Both models have a 512-token context window.

Key class: `OllamaEmbedder(model, host, max_chars)`
Key methods: `embed(texts)`, `embed_one(text)`

---

### `core/store.py`

ChromaDB persistent collection wrapper.

**Ingestion:** pre-computes all embeddings explicitly via `OllamaEmbedder.embed()`
before calling `collection.add()`. This avoids ChromaDB attempting to re-embed
at query time and ensures embedding model consistency.

**Querying with deduplication:** when `deduplicate=True` (default), the store
over-fetches (`top_k * 4` results) then applies `_deduplicate_by_parent()` to
keep only the best-scoring chunk per parent `issue_id`. This ensures that a
long issue with many chunks cannot dominate the results and crowd out other
issues.

Key class: `VectorStore(db_path, collection_name, embedder, batch_size)`
Key methods: `add`, `query(text, top_k, where, deduplicate)`, `count`, `reset`
Key helper: `_deduplicate_by_parent(hits)`

---

### `core/rag.py`

Orchestrates the full RAG loop:

```
extract_filters(question)          ← LLM extracts status/priority filters
        │
        ▼
retrieve(question, store, where)   ← ChromaDB semantic search + deduplication
        │
        ▼
build_prompt(question, retrieved)  ← formats issue context with rich metadata
        │
        ▼
generate(prompt, model)            ← ollama.chat() → answer text
        │
        ▼
return (answer_text, retrieved, filters)
```

**LLM-driven metadata filter extraction:** before retrieval, the question is
sent to the chat LLM with a short extraction prompt asking for JSON
`{"status": "...", "priority": "..."}`. Extracted values are validated against
known Redmine values before being applied as ChromaDB `where=` clauses. This
handles natural-language queries like "show rejected issues" or "high priority
bugs" without requiring explicit CLI flags.

**Graceful fallback:** if filters produce zero results, the query is
automatically retried without filters so the user always gets an answer.

**Rich prompt context:** each retrieved issue is formatted with:
```
[Issue #ID | Project | Status | Priority | created: DATE | updated: DATE]
<full chunk text>
```
The LLM is instructed to cite issue numbers explicitly in its answer.

Key functions: `extract_filters`, `retrieve`, `build_prompt`, `generate`,
`answer(question, store, chat_model, ollama_host, top_k, extract_metadata_filters)`

---

## Pipeline Scripts

### `pipeline/01_download.py`

Downloads all Redmine issues for each configured project.

Flow per project:
1. Load checkpoint if present (resume from interruption)
2. Paginate through all issues via `/issues.json?include=journals`
3. For any issue with empty journals, re-fetch individually via `/issues/<id>.json`
4. Save checkpoint every `SAVE_INTERVAL` issues
5. Write `data/raw/<project_id>.json`

Merge all per-project files into `data/redmine_master.json`.

The adaptive backoff in `RedmineClient` handles rate limiting automatically.

---

### `pipeline/02_anonymize.py`

Runs the two-pass anonymizer over every issue in `data/redmine_master.json`.
Writes `data/redmine_anonymized.json` and `data/user_mapping.json`.
Prints a verification summary including a sample author before/after.

---

### `pipeline/03_ingest.py`

Loads `data/redmine_anonymized.json`, expands each issue into chunks via
`prepare_chunks()`, computes embeddings in batches, and stores everything in
ChromaDB. Reports issue → chunk expansion count and progress per batch.

Supports `--reset` to drop and recreate the collection before ingesting.

---

### `pipeline/04_query.py`

User-facing RAG interface.

- **Single query**: `python pipeline/04_query.py --query "..."`
- **Interactive REPL**: `python pipeline/04_query.py`
- **Show sources**: add `--show-sources` to print retrieved issue metadata

Displays applied metadata filters when the LLM extraction finds any.

---

## Test Strategy

### Unit tests (`tests/unit/`)

210 tests across 7 files. All external I/O is mocked or replaced with
in-memory equivalents.

| Test file | Tests | Key coverage |
|---|---|---|
| `test_document.py` | 43 | `prepare()` and `prepare_chunks()`, truncation, chunking layout |
| `test_anonymizer.py` | 36 | user fields, PII regex (email, IP, hostname), mapping consistency |
| `test_checkpoint.py` | 17 | atomic save, corrupt-file recovery, completion detection |
| `test_redmine_client.py` | 21 | pagination, 404/403/401, adaptive backoff on 429/503 |
| `test_embedder.py` | 11 | batch calls, context cap, Ollama mocked |
| `test_store.py` | 22 | add, query, reset, deduplication by parent issue |
| `test_rag.py` | 39 | filter extraction, JSON parsing, prompt richness, answer() full flow |

### Integration tests (`tests/integration/`)

21 tests in `test_pipeline.py`. Uses 10 synthetic issues and a real in-memory
ChromaDB. Ollama is mocked at the `core.rag.generate` / `core.rag.extract_filters`
level so the test runs without a running server. Covers anonymization →
document preparation → ingestion → retrieval → RAG generation end-to-end.

### Coverage

**96% line coverage** on all `core/` modules (target was 80%).

---

## Implementation History

### Phase 1 — Foundation
- [x] `.env.example`, `requirements.txt`, `config.py`, `core/__init__.py`
- [x] `tests/conftest.py` with shared fixtures
- [x] `.gitignore`

### Phase 2 — Core data modules
- [x] `core/document.py` + `tests/unit/test_document.py`
- [x] `core/anonymizer.py` + `tests/unit/test_anonymizer.py`
- [x] `core/checkpoint.py` + `tests/unit/test_checkpoint.py`

### Phase 3 — Network and storage modules
- [x] `core/redmine_client.py` + `tests/unit/test_redmine_client.py`
- [x] `core/embedder.py` + `tests/unit/test_embedder.py`
- [x] `core/store.py` + `tests/unit/test_store.py`

### Phase 4 — RAG generation
- [x] `core/rag.py` + `tests/unit/test_rag.py`

### Phase 5 — Pipeline scripts
- [x] `pipeline/01_download.py`
- [x] `pipeline/02_anonymize.py`
- [x] `pipeline/03_ingest.py`
- [x] `pipeline/04_query.py`

### Phase 6 — Integration tests and cleanup
- [x] `tests/integration/test_pipeline.py`
- [x] Coverage verified >= 80% (achieved 96%)
- [x] Old scripts removed from repo root

### Phase 7 — Performance and quality improvements

Driven by an interview session identifying concrete improvement areas.

#### 7a — Adaptive backoff in `core/redmine_client.py`

**Problem:** fixed 2s sleep between every request was slow and did not react
to server pressure.

**Solution:** base delay starts at 0.5s. On HTTP 429 or 503, delay doubles
(capped at 60s). On success, delay resets to 0.5s. Goes as fast as the server
allows without risking an IP ban.

**Tests added:** `TestAdaptiveBackoff` — 7 tests covering delay doubling,
cap enforcement, reset on success, retry on 429/503.

---

#### 7b — Free-text PII scan in `core/anonymizer.py`

**Problem:** only structured user objects were anonymized. Descriptions and
journal notes could contain email addresses, IP addresses, and hostnames.

**Solution:** second pass over `description` and journal `notes` using three
compiled regex patterns. Matches are replaced with bracketed placeholders and
logged at DEBUG level.

**Tests added:** `TestScrubPii` — 9 tests; 6 additional tests in
`TestAnonymizeIssue` covering description and journal note scrubbing.

---

#### 7c — Embedding model evaluation

**Finding:** `mxbai-embed-large` (2.6× slower on tested hardware, ~106ms/call)
offers no measurable retrieval quality advantage over `nomic-embed-text`
(~47ms/call) for short-form issue text. Both have a 512-token context window.

**Decision:** retain `nomic-embed-text` as default. `EMBED_MODEL` in `.env`
remains configurable for future experimentation.

---

#### 7d — Section-based chunking in `core/document.py`

**Problem:** one document per issue, truncated to 2000 chars. Long issues with
many journal entries lost most of their history. A question about a resolution
mentioned in comment #8 could not be retrieved.

**Solution:** `prepare_chunks(issue, journals_per_chunk=5, max_text_len=4000)`
splits each issue into:
- Chunk 0 (`issue_<id>`): subject + description
- Chunk N (`issue_<id>_chunk_N`): subject + up to 5 journal entries

561 issues → 1121 chunks. All journal content is now independently retrievable.

**Tests added:** `TestPrepareChunksBasic`, `TestPrepareChunksContent`,
`TestPrepareChunksMetadata` — 20 new tests.

---

#### 7e — Chunk deduplication in `core/store.py`

**Problem:** multiple chunks from the same issue could fill the top-K results,
crowding out other issues.

**Solution:** `query()` over-fetches by 4× then applies `_deduplicate_by_parent()`
which keeps only the best-scoring chunk per parent `issue_id`, returning results
sorted by score.

**Tests added:** `TestDeduplicateByParent` — 5 tests; `test_deduplicate_false_returns_all_chunks`.

---

#### 7f — LLM-driven metadata filter extraction in `core/rag.py`

**Problem:** semantic similarity alone could not distinguish "show rejected
issues" from a general topic query — it returned Resolved issues regardless.

**Solution:** `extract_filters(question, model, host)` sends the question to
the chat LLM with a JSON extraction prompt before retrieval. Extracted
`status` and `priority` values are validated against known Redmine values and
applied as ChromaDB `where=` clauses. If filters produce no results, the query
is automatically retried without them.

**Tests added:** `TestExtractFilters` — 10 tests covering known/unknown values,
API errors, invalid JSON, markdown-fenced responses, empty fallback.

---

#### 7g — Richer prompt context in `core/rag.py`

**Problem:** prompt contained only issue text. The LLM had no date context and
did not reliably cite issue numbers.

**Solution:** issue header in the prompt now includes `created_on` and
`updated_on` timestamps. System prompt instructs the LLM to cite `Issue #ID`
explicitly. `answer()` returns a 3-tuple `(answer_text, retrieved, filters)`
so callers can display which filters were applied.

**Tests added:** `test_created_on_in_prompt`, `test_updated_on_in_prompt`;
all `TestAnswer` tests updated for the new 3-tuple return.

---

#### 7h — Embedder batch efficiency fix

**Problem:** context-window fix (Phase 7c) had changed `embed()` to send one
HTTP call per text. This eliminated the 400 error but added per-call HTTP
overhead back for every chunk.

**Root cause investigation:** measured that `mxbai-embed-large` at ~700ms/call
was the bottleneck, not HTTP overhead. Reverted to `nomic-embed-text` (~47ms)
and fixed `embed()` to send the full list in one batch HTTP call.

**Result:** ingest of 1121 chunks completes in ~5 minutes.

---

### Phase 8 — Observability: timing and progress indicators

Driven by the observation that long-running pipeline steps gave no feedback,
making it impossible to know whether the process was working or stuck.

#### 8a — `core/timing.py` — shared timing module

A new testable module with three tools, no external dependencies:

| Class | Purpose |
|---|---|
| `StageTimer` | Context manager; prints `StageName: 1.2s` on exit; stores `.elapsed` |
| `ProgressBar` | Single-line overwriting bar with throughput and ETA (for non-download stages) |
| `PipelineReport` | Accumulates per-stage timings; prints a summary table with proportional bar chart |

**28 tests** in `tests/unit/test_timing.py`. 100% line coverage.

---

#### 8b — Pipeline scripts instrumented with timing

All four pipeline scripts were updated to import from `core/timing.py` and
report stage-level timing:

**`pipeline/01_download.py`:**
- Every page of the bulk download prints immediately: issue count, %, rate, ETA
- Every individual journal re-fetch prints immediately: issue count, %, rate, ETA
- Summary `PipelineReport` table at the end showing time per project

**`pipeline/02_anonymize.py`:**
- `ProgressBar` during anonymization loop
- Separate `StageTimer` for load, anonymize, and save stages
- `PipelineReport` summary table

**`pipeline/03_ingest.py`:**
- `ProgressBar` during embed+ingest loop with chunks/s throughput and ETA
- Separate `StageTimer` for load, init embedder, init ChromaDB, chunking, embed+ingest
- `PipelineReport` summary table; throughput figure in final summary

**`pipeline/04_query.py`:**
- Per-query breakdown: filter extraction / retrieval / generation / total
- Immediately visible on every query — the key diagnostic for CPU-only hardware

---

#### 8c — Production vs. development mode (`--dev` flag)

**Problem:** running the full pipeline (28k issues, 9 projects) takes hours.
Iterating on code changes required the same multi-hour cycle.

**Solution:** a `--dev` flag on every pipeline script selects a fully isolated
development configuration without touching production data.

**`config.py` additions:**

```
DEV_PROJECT_ID      qesecurity (582 issues, small but real)
DEV_DATA_DIR        ./data/dev  (completely separate from ./data)
DEV_COLLECTION_NAME redmine_issues_dev
```

A `PipelineConfig` dataclass holds all settings. `cfg.prod()` and `cfg.dev()`
return the appropriate snapshot. Pipeline scripts take a `PipelineConfig`
instance — zero `if dev else prod` branching inside the scripts.

**Dev mode per script:**

| Script | `--dev` effect |
|---|---|
| `01_download.py` | Downloads `qesecurity` only into `data/dev/raw/` |
| `02_anonymize.py` | Reads `data/dev/redmine_master.json`, writes `data/dev/redmine_anonymized.json` |
| `03_ingest.py` | Ingests into ChromaDB collection `redmine_issues_dev` in `data/dev/chroma_db/` |
| `04_query.py` | Queries `redmine_issues_dev`; all other behaviour identical to prod |

Additional flags on `04_query.py`:
- `--no-filter` skips the pre-retrieval LLM filter extraction call, saving ~5s
  per query on CPU-only hardware when status/priority filtering is not needed.

**Dev cycle workflow:**

```bash
python pipeline/01_download.py  --dev          # ~30 min for qesecurity
python pipeline/02_anonymize.py --dev          # seconds
python pipeline/03_ingest.py    --dev --reset  # ~5 min
python pipeline/04_query.py     --dev --query "..."
```

Production pipeline is unchanged — no `--dev` flag means full prod config.

---

#### 8d — Progress output reliability fix

**Problem:** `logging.basicConfig` output was buffered when stdout/stderr were
redirected (non-TTY context), causing complete silence for tens of seconds
at the start of the journal re-fetch phase.

**Root cause:** Python's logging system buffers writes when not connected to a
terminal. Additionally, progress inside the journal loop only printed every
`SAVE_INTERVAL` (50) issues — at 0.5s/issue that was 25 seconds of silence.

**Fix:** all progress lines in `pipeline/01_download.py` use
`print(msg, flush=True)` directly, bypassing the logging buffer entirely.
The journal re-fetch now prints one line per issue (one line per ~0.5s),
giving a continuous, visible stream of output regardless of how the script
is invoked.

---

### Phase 9 — Data quality analysis, incremental sync, and production run

#### 9a — openqatests data quality investigation

Before committing to downloading all 13,247 openqatests issues (which would
take ~4 hours), a data quality analysis was performed on the partial dataset.

**Findings:**

| Metric | openqatests | Other projects (avg) |
|---|---|---|
| Auto-generated template (`## Observation`) | 69% | 8–27% |
| Issues with journals (human discussion) | 20% | 90–100% |
| Avg journal entries/issue | 1.5 | 5–7 |
| Contains openQA URL links | 73% | rare |

**Decision:** download openqatests but apply a journals-only filter — keep
only the ~2,600 issues that have at least one human comment. This is
controlled by `JOURNALS_ONLY_PROJECTS=openqatests` in `.env` and the
`config.JOURNALS_ONLY_PROJECTS` set in `config.py`. The filter is applied
at the end of `download_project_full()` after all journals are fetched.

---

#### 9b — Incremental sync (`--since` / `--sync`)

**Problem:** the dataset needs regular updates as new Redmine issues are
created and existing ones receive new comments. Re-downloading all 28k
issues every time is not feasible.

**Solution:** `pipeline/01_download.py` gains two new flags:

```
--since DATE   Fetch only issues with updated_on >= DATE.
               Merges into existing per-project JSON files.
--sync         Use the date from the last successful sync automatically.
               Equivalent to --since <last_synced_at>.
```

Using `updated_on >= SINCE` captures both new issues (also have
`updated_on >= SINCE`) and existing issues that received new comments —
one filter covers everything needed for a complete incremental sync.

**New in `core/checkpoint.py`:**
- `save_sync_state(raw_dir, project_id, synced_at)` — records the timestamp
  of each successful sync per project
- `get_last_synced_at(raw_dir, project_id)` — reads it back for `--sync`

**New in `core/redmine_client.py`:**
- `fetch_updated_since(project_id, since, offset, limit)` — paginated fetch
  with `updated_on >= SINCE` filter

**New in `pipeline/01_download.py`:**
- `merge_into(existing, incoming)` — merges new/updated issues into the
  existing id→issue map; returns (n_new, n_updated) counts
- `sync_project(client, c, project_id, since)` — full incremental sync
  for one project

**Sync state files:** `data/raw/<project_id>_sync.json` — written after
each successful run with the timestamp of when the sync started (conservative:
uses start time, not end time, to prevent gaps between consecutive runs).

**11 new tests** in `test_checkpoint.py`, **6 new tests** in
`test_redmine_client.py`, **6 new tests** in `test_download.py`.

---

#### 9c — Crash-resistant journal re-fetch

**Problem:** the journal re-fetch loop crashed the entire pipeline on any
network error (DNS failure, timeout), losing all progress for the current
project.

**Fix:** the journal loop now catches all `Exception` types per issue,
logs the error, saves progress, and continues to the next issue. After 3
consecutive failures it pauses for 30 seconds before continuing. This
allows the download to survive transient network outages without losing work.

---

#### 9d — Upsert instead of add in ChromaDB

**Problem:** `collection.add()` raises `DuplicateIDError` if the ingest is
interrupted and restarted, making it impossible to resume without `--reset`
(which discards all already-embedded chunks).

**Fix:** `core/store.py` switched from `collection.add()` to
`collection.upsert()`. Upsert inserts new documents and updates existing
ones with no error on duplicates, making the ingest fully resumable at any
point without data loss.

---

#### 9e — Full production download and first demo

**Data collected (2026-05-29):**

| Project | Issues | With journals | Journal entries |
|---|---|---|---|
| virtualization | 2,778 | ~2,700 | ~21,000 |
| performance | 2,076 | ~1,950 | ~10,500 |
| qesecurity | 765 | 748 | 4,969 |
| qe-kernel | 1,867 | 1,858 | 13,241 |
| qam | 561 | ~450 | ~2,700 |
| qe-yast | 2,429 | 2,409 | 21,392 |
| openqatests | 13,247 | 7,593 (filtered) | 57,150 |
| openqa-infrastructure | downloading | — | — |
| containers | downloading | — | — |

**Master dataset (partial, demo):** 18,069 issues → 40,509 chunks

**Demo:** successfully presented to a group of QA engineers using the
partial collection (13,700 chunks, 4 projects). Queries demonstrated:
- Semantic search finding s390x issues across projects
- Automatic `status: Rejected` filter extraction from natural language
- Cross-project synthesis of security vulnerability history
- Engineering knowledge extraction from 276-comment Agama epic

**Key hardware finding:** on CPU-only hardware, `nomic-embed-text` embeds
at ~2.8 chunks/s (limited by the embedding model's matrix operations, not
I/O). Ingesting 40,509 chunks takes ~4 hours. LLM generation (llama3) takes
25–55 seconds per query. Both are fundamental hardware constraints with no
software workaround beyond acquiring a GPU.

**Process management:** all long-running jobs are managed via `screen`
sessions (`redmine-download`, `redmine-ingest`) with output to log files
in `logs/`. The ingest is fully resumable via upsert.

---

### Phase 10 — Design interview decisions (pending implementation)

A systematic design interview resolved all open architectural questions.
The following changes are planned for implementation:

#### Data pipeline
- **openQA URL stripping:** strip `openqa.suse.de` and `openqa.opensuse.org`
  URLs from description and journal text before embedding (add to
  `core/anonymizer.py` as a new regex pattern). The URLs add noise to
  embeddings with no semantic value.
- **Score threshold:** `store.query()` to return an empty list when all
  results exceed a minimum L2 distance threshold, rather than always
  returning top-K regardless of quality.

#### Retrieval
- **Project metadata filter:** extend `rag.extract_filters()` to also
  extract project names from natural language ("qe-kernel issues",
  "containers bugs") and apply as a `project_id` ChromaDB filter alongside
  status and priority.
- **Stronger citation instruction:** update the system prompt to require
  `(Issue #ID)` after every claim, not just suggest it.

#### Interface
- **Batch query mode:** `pipeline/04_query.py --queries-file questions.txt`
  reads questions one per line and outputs answers to stdout or a file.
  Useful for benchmarking retrieval quality.
- **Background download flag:** `pipeline/01_download.py --background`
  daemonizes the process and writes to `logs/download.log` automatically,
  eliminating the need to manually set up `nohup` or `screen`.

#### Evaluation
- **`tests/eval/` framework:** a curated `questions.jsonl` with
  `{question, expected_issue_ids[]}` entries. A hit-rate script runs each
  question against the live collection and reports what fraction of expected
  issues appear in top-K results.

#### Documentation
- **`README.md`:** full setup guide, all pipeline steps, all flags,
  troubleshooting section. Target: a new team member gets from zero to
  working queries by following the README alone.

---

## Dependencies

```
# Redmine API
requests>=2.31
python-dotenv>=1.0

# Vector store
chromadb>=0.5

# Embeddings and LLM
ollama>=0.2

# Testing
pytest>=8.0
pytest-mock>=3.14
responses>=0.25
pytest-cov>=5.0
```

---

## Running the Pipeline

```bash
# Setup
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in REDMINE_API_KEY, PROJECT_IDS, etc.

# Full production pipeline
python pipeline/01_download.py               # adaptive rate limit, crash-safe
python pipeline/02_anonymize.py              # seconds
python pipeline/03_ingest.py --reset         # chunks + embeds (~4h for 18k issues on CPU)

# Incremental sync (after first full run)
python pipeline/01_download.py --sync        # fetches only issues updated since last run
python pipeline/02_anonymize.py
python pipeline/03_ingest.py                 # upsert — safe to run without --reset

# Dev mode (isolated, fast cycle)
python pipeline/01_download.py  --dev        # qesecurity only, data/dev/
python pipeline/02_anonymize.py --dev
python pipeline/03_ingest.py    --dev --reset
python pipeline/04_query.py     --dev --query "..."

# Query
python pipeline/04_query.py --query "Are there rejected issues and why?"
python pipeline/04_query.py --query "..." --show-sources
python pipeline/04_query.py --query "..." --no-filter   # skip LLM filter (~5s faster on CPU)
python pipeline/04_query.py                              # interactive REPL

# Monitor long-running background jobs
screen -ls                                   # list screen sessions
tail -f logs/download.log                    # live download progress
tail -f logs/ingest.log | grep -v HTTP       # live ingest progress
```

---

## Known Limitations and Future Work

| Area | Current state | Planned / potential improvement |
|---|---|---|
| Streaming output | Blocking `ollama.chat()` — silent until done | `ollama.chat(stream=True)` for token-by-token output |
| Conversation memory | Stateless — each query is independent | Add last-N-turns to `messages[]` for follow-up questions |
| openQA URL noise | URLs embedded verbatim | Strip `openqa.suse.de` / `openqa.opensuse.org` URLs in anonymizer |
| Score threshold | Always returns top-K even on poor matches | Return empty result + "no relevant issues" when all scores exceed threshold |
| Project filter | Only status/priority extracted from NL | Add project name extraction to `extract_filters()` |
| Batch query mode | Single query or REPL only | `--queries-file` flag for automated benchmarking |
| Background download | Requires manual `screen` / `nohup` setup | `--background` flag to daemonize + log automatically |
| Evaluation framework | Manual spot-checking only | `tests/eval/questions.jsonl` + hit-rate script |
| README | Not written | Full setup → pipeline → flags → troubleshooting guide |
| GPU / response time | ~30s generation on CPU | GPU or API-based LLM for 2–5s responses |
| Web UI | CLI only | FastAPI + minimal HTML frontend |
| Data completeness | 7/9 projects ingested | openqa-infrastructure + containers still downloading |
