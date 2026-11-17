"""Budget complete evidence chunks and construct role-separated RAG messages."""

from functools import cache
from importlib.resources import files
import json
from uuid import UUID

import tiktoken

from app.rag.schemas import (
    PreparedPrompt, PromptConfig, PromptMessage, RetrievedChunk, SkippedContext,
)


PROMPT_VERSION = "rag-v1"


class PromptInputError(ValueError):
    """The user's question cannot fit the configured RAG input policy."""


@cache
def _system_prompt() -> str:
    return files("app.rag").joinpath("templates/system.md").read_text(encoding="utf-8").strip()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_context_source(chunk: RetrievedChunk, label: str) -> dict:
    """Whitelist useful context; do not blindly expose all stored metadata."""
    return {
        "label": label,
        "chunk_id": str(chunk.chunk_id),
        "title": chunk.title,
        "source_path": chunk.source_path,
        "version_number": chunk.version_number,
        "location": {
            "heading_path": chunk.metadata.get("heading_path", []),
            "page_number": chunk.metadata.get("page_number"),
            "source_line_ranges": chunk.metadata.get("source_line_ranges", []),
        },
        "content": chunk.content,
    }


def build_rag_prompt(
    question: str, chunks: list[RetrievedChunk], *, config: PromptConfig | None = None,
) -> PreparedPrompt:
    """Prepare messages only: no model calls, ranking changes or DB writes.

    S1/S2 labels are local to this prompt. Citation mapping must follow this
    selected order, never the unfiltered retrieval order.
    Token counts estimate message content only; model-specific framing and
    generation-output allowance must be handled by the future Chat Provider.
    """
    config = config or PromptConfig()
    question = question.strip()
    if not question:
        raise PromptInputError("Question must not be empty")
    encoding = tiktoken.get_encoding(config.tokenizer)

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    if count(question) > config.max_question_tokens:
        raise PromptInputError("Question exceeds the configured prompt token budget")
    system = _system_prompt()
    system_tokens = count(system)
    sources: list[dict] = []
    user = _json({"question": question, "sources": sources})
    if system_tokens + count(user) > config.max_input_tokens:
        raise ValueError("Input token budget is too small for the system rules and question")

    selected: list[UUID] = []
    seen: set[UUID] = set()
    skipped: list[SkippedContext] = []
    for chunk in chunks:
        reason = None
        if chunk.chunk_id in seen:
            reason = "duplicate"
        elif not chunk.content.strip():
            reason = "empty_content"
        elif len(sources) >= config.max_sources:
            reason = "source_limit"
        seen.add(chunk.chunk_id)
        if reason is None:
            candidate = [*sources, render_context_source(chunk, f"S{len(sources) + 1}")]
            candidate_user = _json({"question": question, "sources": candidate})
            if count(_json(candidate)) > config.max_context_tokens:
                reason = "context_budget"
            elif system_tokens + count(candidate_user) > config.max_input_tokens:
                reason = "input_budget"
            else:
                sources = candidate
                user = candidate_user
                selected.append(chunk.chunk_id)
        if reason is not None:
            skipped.append(SkippedContext(chunk_id=chunk.chunk_id, reason=reason))

    return PreparedPrompt(
        prompt_version=PROMPT_VERSION,
        messages=[PromptMessage(role="system", content=system), PromptMessage(role="user", content=user)],
        selected_chunk_ids=selected, skipped_context=skipped,
        estimated_context_tokens=count(_json(sources)),
        estimated_input_tokens=system_tokens + count(user), tokenizer=config.tokenizer, config=config,
    )
