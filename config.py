"""
Central configuration for the Redmine RAG pipeline.

All tuneable values live here. Every other module imports from this file.
Values are read from environment variables (populated from a .env file via
python-dotenv). Sensible defaults are provided for optional settings.

Production vs. development mode
--------------------------------
Pipeline scripts accept a ``--dev`` flag. When active, they call
``config.dev()`` which returns a ``DevConfig`` snapshot — a copy of the
production config with the following overrides:

  - PROJECT_IDS  → [DEV_PROJECT_ID]  (single small project)
  - DATA_DIR     → DEV_DATA_DIR      (isolated from prod data)
  - RAW_DIR      → DEV_DATA_DIR/raw
  - MASTER_FILE  → DEV_DATA_DIR/redmine_master.json
  - ANONYMIZED_FILE → DEV_DATA_DIR/redmine_anonymized.json
  - USER_MAPPING_FILE → DEV_DATA_DIR/user_mapping.json
  - CHROMA_DIR   → DEV_DATA_DIR/chroma_db
  - COLLECTION_NAME → DEV_COLLECTION_NAME

All other values (API key, models, batch size, etc.) are identical to prod.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Redmine connection
# ---------------------------------------------------------------------------
REDMINE_BASE_URL: str = os.getenv("REDMINE_BASE_URL", "https://progress.opensuse.org")
REDMINE_API_KEY: str = os.getenv("REDMINE_API_KEY", "")

_raw_project_ids = os.getenv("PROJECT_IDS", "")
PROJECT_IDS: list[str] = [p.strip() for p in _raw_project_ids.split(",") if p.strip()]

# ---------------------------------------------------------------------------
# Download behaviour
# ---------------------------------------------------------------------------
RATE_LIMIT_SECONDS: float = float(os.getenv("RATE_LIMIT_SECONDS", "2.0"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
SAVE_INTERVAL: int = int(os.getenv("SAVE_INTERVAL", "50"))
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR: Path = Path(os.getenv("RAW_DIR", str(DATA_DIR / "raw")))
MASTER_FILE: Path = Path(os.getenv("MASTER_FILE", str(DATA_DIR / "redmine_master.json")))
ANONYMIZED_FILE: Path = Path(os.getenv("ANONYMIZED_FILE", str(DATA_DIR / "redmine_anonymized.json")))
USER_MAPPING_FILE: Path = Path(os.getenv("USER_MAPPING_FILE", str(DATA_DIR / "user_mapping.json")))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", str(DATA_DIR / "chroma_db")))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "redmine_issues")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama3.2")

# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "50"))
MAX_TEXT_LEN: int = int(os.getenv("MAX_TEXT_LEN", "8192"))

# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
TOP_K: int = int(os.getenv("TOP_K", "5"))
# Maximum L2 distance to accept from the vector store.  Results whose best
# score exceeds this threshold are treated as "no relevant match found".
# None disables the check.  Typical nomic-embed-text L2 range: 0.0–2.0.
# A value around 1.2 rejects clearly unrelated queries while keeping
# borderline relevant results.
_raw_score_threshold = os.getenv("SCORE_THRESHOLD", "")
SCORE_THRESHOLD: float | None = float(_raw_score_threshold) if _raw_score_threshold else None

# ---------------------------------------------------------------------------
# Per-project filters
# ---------------------------------------------------------------------------
# Comma-separated list of project IDs for which only issues that have at
# least one journal entry should be downloaded and ingested.
# Useful for auto-generated projects like openqatests where most issues
# have no human commentary and add noise rather than signal.
_raw_journals_only = os.getenv("JOURNALS_ONLY_PROJECTS", "openqatests")
JOURNALS_ONLY_PROJECTS: set[str] = {
    p.strip() for p in _raw_journals_only.split(",") if p.strip()
}

# ---------------------------------------------------------------------------
# Development mode overrides
# ---------------------------------------------------------------------------
DEV_PROJECT_ID: str = os.getenv("DEV_PROJECT_ID", "qesecurity")
DEV_DATA_DIR: Path = Path(os.getenv("DEV_DATA_DIR", "./data/dev"))
DEV_COLLECTION_NAME: str = os.getenv("DEV_COLLECTION_NAME", "redmine_issues_dev")


# ---------------------------------------------------------------------------
# Dev config factory
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    A snapshot of all pipeline configuration values.

    Pipeline scripts work with a PipelineConfig instance (either the
    production singleton or the dev snapshot returned by ``dev()``) so
    that --dev mode needs zero conditional logic in the scripts themselves.
    """
    # Redmine
    REDMINE_BASE_URL: str
    REDMINE_API_KEY: str
    PROJECT_IDS: list
    RATE_LIMIT_SECONDS: float
    MAX_RETRIES: int
    SAVE_INTERVAL: int
    REQUEST_TIMEOUT: int
    # Paths
    DATA_DIR: Path
    RAW_DIR: Path
    MASTER_FILE: Path
    ANONYMIZED_FILE: Path
    USER_MAPPING_FILE: Path
    CHROMA_DIR: Path
    COLLECTION_NAME: str
    # Ollama
    OLLAMA_HOST: str
    EMBED_MODEL: str
    CHAT_MODEL: str
    # Ingest
    BATCH_SIZE: int
    MAX_TEXT_LEN: int
    # Query
    TOP_K: int
    SCORE_THRESHOLD: float | None
    # Per-project filters
    JOURNALS_ONLY_PROJECTS: set
    # Mode
    is_dev: bool = False

    def label(self) -> str:
        """Human-readable mode label for log output."""
        return "DEV" if self.is_dev else "PROD"


def prod() -> PipelineConfig:
    """Return the production PipelineConfig (reads module-level variables)."""
    return PipelineConfig(
        REDMINE_BASE_URL=REDMINE_BASE_URL,
        REDMINE_API_KEY=REDMINE_API_KEY,
        PROJECT_IDS=PROJECT_IDS,
        RATE_LIMIT_SECONDS=RATE_LIMIT_SECONDS,
        MAX_RETRIES=MAX_RETRIES,
        SAVE_INTERVAL=SAVE_INTERVAL,
        REQUEST_TIMEOUT=REQUEST_TIMEOUT,
        DATA_DIR=DATA_DIR,
        RAW_DIR=RAW_DIR,
        MASTER_FILE=MASTER_FILE,
        ANONYMIZED_FILE=ANONYMIZED_FILE,
        USER_MAPPING_FILE=USER_MAPPING_FILE,
        CHROMA_DIR=CHROMA_DIR,
        COLLECTION_NAME=COLLECTION_NAME,
        OLLAMA_HOST=OLLAMA_HOST,
        EMBED_MODEL=EMBED_MODEL,
        CHAT_MODEL=CHAT_MODEL,
        BATCH_SIZE=BATCH_SIZE,
        MAX_TEXT_LEN=MAX_TEXT_LEN,
        TOP_K=TOP_K,
        SCORE_THRESHOLD=SCORE_THRESHOLD,
        JOURNALS_ONLY_PROJECTS=JOURNALS_ONLY_PROJECTS,
        is_dev=False,
    )


def dev() -> PipelineConfig:
    """
    Return a dev-mode PipelineConfig with isolated paths and a single
    small project.  All Redmine credentials and Ollama settings are
    inherited from the production config.
    """
    dev_dir = DEV_DATA_DIR
    return PipelineConfig(
        REDMINE_BASE_URL=REDMINE_BASE_URL,
        REDMINE_API_KEY=REDMINE_API_KEY,
        PROJECT_IDS=[DEV_PROJECT_ID],
        RATE_LIMIT_SECONDS=RATE_LIMIT_SECONDS,
        MAX_RETRIES=MAX_RETRIES,
        SAVE_INTERVAL=SAVE_INTERVAL,
        REQUEST_TIMEOUT=REQUEST_TIMEOUT,
        DATA_DIR=dev_dir,
        RAW_DIR=dev_dir / "raw",
        MASTER_FILE=dev_dir / "redmine_master.json",
        ANONYMIZED_FILE=dev_dir / "redmine_anonymized.json",
        USER_MAPPING_FILE=dev_dir / "user_mapping.json",
        CHROMA_DIR=dev_dir / "chroma_db",
        COLLECTION_NAME=DEV_COLLECTION_NAME,
        OLLAMA_HOST=OLLAMA_HOST,
        EMBED_MODEL=EMBED_MODEL,
        CHAT_MODEL=CHAT_MODEL,
        BATCH_SIZE=BATCH_SIZE,
        MAX_TEXT_LEN=MAX_TEXT_LEN,
        TOP_K=TOP_K,
        SCORE_THRESHOLD=SCORE_THRESHOLD,
        JOURNALS_ONLY_PROJECTS=JOURNALS_ONLY_PROJECTS,
        is_dev=True,
    )
