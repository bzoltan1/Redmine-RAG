"""
core/rag.py — Retrieval-Augmented Generation orchestration.

Implements the full RAG loop:
  1. Extract metadata filters from the question using the LLM.
  2. Retrieve the top-K most relevant issues from the vector store,
     applying any extracted filters and deduplicating by parent issue.
  3. Build a grounded prompt with rich issue context (dates, priority, citations).
  4. Send the prompt to an Ollama chat model for answer generation.
  5. Return the generated answer text and the retrieved sources.

No file I/O. No CLI logic.
"""

import json
import logging
from typing import Any

from ollama import Client

from core.store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known filterable metadata values (must match what is stored in ChromaDB)
# ---------------------------------------------------------------------------

_KNOWN_STATUSES   = {"New", "In Progress", "Resolved", "Rejected", "Closed", "Feedback"}
_KNOWN_PRIORITIES = {"Low", "Normal", "High", "Urgent", "Immediate"}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a Redmine issue tracker database. "
    "Use the retrieved issues below to answer the user's question as accurately as possible. "
    "Cite specific issue numbers (e.g. 'Issue #1234') when referring to retrieved issues. "
    "If the retrieved issues do not contain enough information, say so clearly rather than guessing."
)

_ISSUE_TEMPLATE = (
    "[Issue #{issue_id} | {project} | {status} | {priority} | "
    "created: {created_on} | updated: {updated_on}]\n"
    "{text}\n"
)

_PROMPT_TEMPLATE = """\
--- Retrieved Issues ---
{issues_block}
--- End of Issues ---

Question: {question}
"""

_FILTER_EXTRACTION_PROMPT = """\
Extract any Redmine metadata filters from the following question.
Return a JSON object with these optional keys:
  "status"   - one of: New, In Progress, Resolved, Rejected, Closed, Feedback
  "priority" - one of: Low, Normal, High, Urgent, Immediate

Only include a key if you are confident the question is asking to filter by that value.
If no filters apply, return an empty JSON object: {{}}

Examples:
  "Show me rejected issues"       -> {{"status": "Rejected"}}
  "Any high priority bugs?"       -> {{"priority": "High"}}
  "What is the team working on?"  -> {{}}

Question: {question}

Respond with only the JSON object, no explanation.
"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_filters(
    question: str,
    model: str,
    host: str = "http://localhost:11434",
) -> dict[str, str]:
    """
    Ask the LLM to extract ChromaDB metadata filters from *question*.

    Returns a dict suitable for use as a ChromaDB ``where=`` clause,
    e.g. {"status": "Rejected"} or {} if no filters were found.

    Falls back to {} on any parse or API error so retrieval always proceeds.
    """
    client = Client(host=host)
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "user", "content": _FILTER_EXTRACTION_PROMPT.format(question=question)},
            ],
        )
        raw = response.message.content if hasattr(response, "message") else response["message"]["content"]
        raw = raw.strip()
        # Strip markdown code fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        filters: dict = json.loads(raw)
    except Exception as exc:
        logger.debug("Filter extraction failed (%s), proceeding without filters.", exc)
        return {}

    # Validate: only keep known values to avoid broken ChromaDB queries
    validated: dict[str, str] = {}
    if "status" in filters and filters["status"] in _KNOWN_STATUSES:
        validated["status"] = filters["status"]
    if "priority" in filters and filters["priority"] in _KNOWN_PRIORITIES:
        validated["priority"] = filters["priority"]

    if validated:
        logger.info("Extracted filters: %s", validated)
    return validated


def retrieve(
    question: str,
    store: VectorStore,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Perform semantic search against the vector store.

    Parameters
    ----------
    question: The user's natural-language question.
    store:    Initialised VectorStore instance.
    top_k:    Number of unique parent issues to retrieve.
    where:    Optional ChromaDB metadata filter dict.

    Returns
    -------
    List of result dicts from VectorStore.query(), deduplicated by parent issue.
    """
    return store.query(question, top_k=top_k, where=where or None, deduplicate=True)


def build_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
    """
    Assemble the RAG prompt from the user's question and retrieved issues.

    Includes rich metadata (dates, priority) and instructs the LLM to
    cite issue numbers explicitly.
    """
    issue_blocks: list[str] = []
    for item in retrieved:
        meta = item.get("metadata") or {}
        block = _ISSUE_TEMPLATE.format(
            issue_id=  meta.get("issue_id",  "?"),
            project=   meta.get("project",   "?"),
            status=    meta.get("status",    "?"),
            priority=  meta.get("priority",  "?"),
            created_on=meta.get("created_on","?"),
            updated_on=meta.get("updated_on","?"),
            text=      item.get("text", "").strip(),
        )
        issue_blocks.append(block)

    issues_block = "\n---\n".join(issue_blocks) if issue_blocks else "(no issues retrieved)"

    return _PROMPT_TEMPLATE.format(
        issues_block=issues_block,
        question=question,
    )


def generate(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
) -> str:
    """
    Send the prompt to an Ollama chat model and return the answer text.
    """
    client = Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def answer(
    question: str,
    store: VectorStore,
    chat_model: str,
    ollama_host: str = "http://localhost:11434",
    top_k: int = 5,
    extract_metadata_filters: bool = True,
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    """
    Full RAG pipeline: extract filters → retrieve → build prompt → generate.

    Parameters
    ----------
    question:                 The user's question.
    store:                    Initialised VectorStore.
    chat_model:               Ollama chat model name.
    ollama_host:              Ollama server URL.
    top_k:                    Number of unique issues to retrieve.
    extract_metadata_filters: If True, ask the LLM to extract status/priority
                              filters from the question before retrieval.

    Returns
    -------
    (answer_text, retrieved_issues, applied_filters) tuple.
    """
    filters: dict[str, str] = {}
    if extract_metadata_filters:
        filters = extract_filters(question, model=chat_model, host=ollama_host)

    # Build ChromaDB where= clause from filters
    where: dict | None = None
    if filters:
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            where = {key: {"$eq": val}}
        else:
            where = {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    retrieved = retrieve(question, store, top_k=top_k, where=where)

    # If filters produced no results, retry without them
    if not retrieved and where is not None:
        logger.info("Filter produced no results; retrying without filter.")
        retrieved = retrieve(question, store, top_k=top_k, where=None)
        filters = {}

    prompt = build_prompt(question, retrieved)
    answer_text = generate(prompt, model=chat_model, host=ollama_host)
    return answer_text, retrieved, filters
