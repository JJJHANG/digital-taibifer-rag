"""Validate citation identities and map them only to selected backend sources."""

from collections.abc import Mapping
import json
import re

from markdown_it import MarkdownIt
from pydantic import ValidationError

from app.rag.prompt import render_context_source
from app.rag.schemas import (
    ChatUsage, CitationReport, CitationSource, PreparedPrompt, RetrievedChunk,
)


MARKER = re.compile(r"\[[ \t]*[Ss][^\[\]\r\n]*\]")
CANONICAL_MARKER = re.compile(r"\[S[1-9][0-9]*\]")
RESERVED_LINK_LABEL = re.compile(r"[Ss][0-9]+")


class CitationContextError(ValueError):
    """Prompt, selected identities, or backend source metadata are inconsistent."""


class CitationValidationError(RuntimeError):
    """The answer used unknown/malformed citations; do not publish it as valid."""

    def __init__(self, report: CitationReport, *, usage: ChatUsage | None = None) -> None:
        super().__init__("invalid_citation_markers")
        self.report = report
        self.usage = usage


def build_citation_map(
    prompt: PreparedPrompt, chunks: list[RetrievedChunk],
) -> dict[str, CitationSource]:
    """Verify actual prompt evidence, then snapshot sources before generation."""
    by_id: dict = {}
    for chunk in chunks:
        if chunk.chunk_id in by_id and by_id[chunk.chunk_id] != chunk:
            raise CitationContextError("conflicting_chunk_identity")
        by_id.setdefault(chunk.chunk_id, chunk)
    if len(set(prompt.selected_chunk_ids)) != len(prompt.selected_chunk_ids):
        raise CitationContextError("duplicate_selected_chunk_identity")
    if [message.role for message in prompt.messages] != ["system", "user"]:
        raise CitationContextError("invalid_prompt_roles")
    try:
        selected = [by_id[chunk_id] for chunk_id in prompt.selected_chunk_ids]
        actual_sources = json.loads(prompt.messages[1].content)["sources"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise CitationContextError("selected_context_missing") from None
    expected = [render_context_source(chunk, f"S{index}") for index, chunk in enumerate(selected, start=1)]
    if actual_sources != expected:
        raise CitationContextError("prompt_source_content_mismatch")

    sources: dict[str, CitationSource] = {}
    try:
        for index, chunk in enumerate(selected, start=1):
            label = f"S{index}"
            sources[label] = CitationSource(
                citation_id=label, chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                document_version_id=chunk.document_version_id, version_number=chunk.version_number,
                chunk_index=chunk.chunk_index, title=chunk.title, source_path=chunk.source_path,
                source_type=chunk.source_type,
                source_url=chunk.metadata.get("document_metadata", {}).get("source_url"),
                heading_path=chunk.metadata.get("heading_path", []),
                page_number=chunk.metadata.get("page_number"),
                source_line_ranges=chunk.metadata.get("source_line_ranges", []),
            )
    except (ValidationError, AttributeError, TypeError):
        raise CitationContextError("source_metadata_invalid") from None
    return sources


def _markers(answer: str) -> list[str]:
    """Read prose markers; code examples are not evidence citations.

    Reserved S-number link labels are invalid citation syntax. URLs are never
    read from model output to construct the authoritative source list.
    """
    found: list[str] = []
    for block in MarkdownIt("commonmark").parse(answer):
        if block.type != "inline":
            continue
        link_depth = 0
        for token in block.children or []:
            if token.type == "link_open":
                link_depth += 1
            elif token.type == "link_close":
                link_depth -= 1
            elif token.type == "text":
                if link_depth and RESERVED_LINK_LABEL.fullmatch(token.content):
                    found.append(f"[{token.content}](link)")
                else:
                    found.extend(match.group() for match in MARKER.finditer(token.content))
    return list(dict.fromkeys(found))


def validate_citations(
    answer: str, source_map: Mapping[str, CitationSource], *, expected: bool = True,
) -> CitationReport:
    """Check format/existence only, not factual entailment or claim coverage.

    Missing markers are reported explicitly: an uncited factual answer and a
    knowledge-insufficiency refusal cannot be distinguished reliably by regex.
    """
    if any(label != source.citation_id for label, source in source_map.items()):
        raise CitationContextError("citation_map_key_mismatch")
    markers = _markers(answer)
    if not markers:
        return CitationReport(status="missing" if expected else "not_applicable")
    invalid: list[str] = []
    cited_ids: list[str] = []
    for marker in markers:
        if CANONICAL_MARKER.fullmatch(marker) is None:
            invalid.append(marker)
            continue
        label = marker[1:-1]
        cited_ids.append(label)
        if label not in source_map:
            invalid.append(marker)
    return CitationReport(
        status="invalid" if invalid else "valid", cited_ids=cited_ids,
        invalid_markers=invalid,
        sources=[] if invalid else [source_map[label] for label in cited_ids],
    )
