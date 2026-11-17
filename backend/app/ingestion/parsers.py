"""Parse text PDFs, Markdown, and UTF-8 TXT without indexing side effects.

Raw Markdown is retained, including code blocks and tables. Tokenization is
used only to locate real top-level headings and build their hierarchy.
"""

from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.errors import PyPdfError
import yaml

from app.ingestion.schemas import (
    LineRange,
    ParsedDocument,
    ParsedSection,
    ParseWarning,
)


class ParseError(ValueError):
    """Invalid, unsupported, or unreadable document content."""


def _clean_text(text: str) -> str:
    # Preserve indentation, paragraph boundaries, and meaningful Unicode.
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def _source_ranges(start: int, end: int, excluded: list[LineRange]) -> list[LineRange]:
    """Map prepared-file lines back across excluded original FAQ ranges."""
    ranges: list[LineRange] = []
    for line in range(start, end + 1):
        original = line
        for gap in excluded:
            if original < gap.start:
                break
            original += gap.end - gap.start + 1
        if ranges and original == ranges[-1].end + 1:
            ranges[-1].end = original
        else:
            ranges.append(LineRange(start=original, end=original))
    return ranges


def _text_section(
    lines: list[str], start: int, end: int, headings: list[str], excluded: list[LineRange]
) -> ParsedSection | None:
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    if start == end:
        return None
    return ParsedSection(
        content="".join(lines[start:end]).rstrip("\n"),
        heading_path=headings,
        line_range=LineRange(start=start + 1, end=end),
        source_line_ranges=_source_ranges(start + 1, end, excluded),
    )


def _markdown_sections(
    text: str, excluded: list[LineRange]
) -> tuple[list[ParsedSection], dict[str, Any], str | None]:
    lines = text.splitlines(keepends=True)
    front_matter: dict[str, Any] = {}
    if lines and lines[0].strip() == "---":
        stop = next(
            (i for i in range(1, len(lines)) if lines[i].strip() in {"---", "..."}), None
        )
        if stop is None:
            raise ParseError("Markdown front matter is not closed")
        try:
            loaded = yaml.safe_load("".join(lines[1:stop]))
        except yaml.YAMLError as error:
            raise ParseError("Invalid YAML front matter") from error
        if loaded is not None and (
            not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded)
        ):
            raise ParseError("Markdown front matter must be a string-keyed mapping")
        front_matter = loaded or {}
        # Blank rather than delete, so token coordinates still match input.
        lines[:stop + 1] = ["\n"] * (stop + 1)

    tokens = MarkdownIt("commonmark").parse("".join(lines))
    sections: list[ParsedSection] = []
    stack: list[tuple[int, str]] = []
    cursor = 0
    first_heading = None
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.level != 0 or token.map is None:
            continue
        start = token.map[0]
        section = _text_section(lines, cursor, start, [label for _, label in stack], excluded)
        if section is not None:
            sections.append(section)
        inline = tokens[index + 1]
        label = "".join(
            child.content for child in (inline.children or [])
            if child.type in {"text", "code_inline", "image", "softbreak", "hardbreak"}
        ).strip() or inline.content
        level = int(token.tag[1:])
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, label))
        if first_heading is None:
            first_heading = label
        cursor = start
    section = _text_section(lines, cursor, len(lines), [label for _, label in stack], excluded)
    if section is not None:
        sections.append(section)
    return sections, front_matter, first_heading


def _pdf_sections(path: Path) -> tuple[list[ParsedSection], list[ParseWarning]]:
    sections: list[ParsedSection] = []
    warnings: list[ParseWarning] = []
    try:
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise ParseError("Encrypted PDFs are not supported; supply an unencrypted copy")
            for page_number, page in enumerate(reader.pages, start=1):
                content = _clean_text(page.extract_text() or "").strip()
                if content:
                    sections.append(ParsedSection(content=content, page_number=page_number))
                else:
                    warnings.append(ParseWarning(
                        code="page_without_text",
                        page_number=page_number,
                        message="No extractable text: blank, image-only, or unsupported page; review required",
                    ))
    except (PyPdfError, OSError) as error:
        raise ParseError("PDF could not be read or its text could not be extracted") from error
    if not sections:
        raise ParseError("PDF has no extractable text; review for blank pages, scanning/OCR, or unsupported encoding")
    return sections, warnings


def parse_document(
    path: Path,
    *,
    document_id: str,
    title: str | None = None,
    source_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    excluded_line_ranges: list[LineRange] | None = None,
) -> ParsedDocument:
    """Parse one deliberately selected file; never scan a directory or a FAQ set.

    ``path`` is the actual input. ``source_path`` identifies the original source.
    Pass Manifest faq_line_ranges as excluded_line_ranges for prepared Markdown.
    Omitting these ranges means the input's line numbers are the original ones.
    PDF page numbers are one-based physical pages, not printed page labels.
    """
    path = Path(path)
    source_type = {".pdf": "pdf", ".md": "markdown", ".txt": "txt"}.get(path.suffix.lower())
    if source_type is None:
        raise ParseError(f"Unsupported file extension: {path.suffix or '(none)'}")
    excluded = list(excluded_line_ranges or [])
    if excluded and source_type != "markdown":
        raise ParseError("Excluded FAQ line ranges are only supported for prepared Markdown")
    if any(current.start <= previous.end for previous, current in zip(excluded, excluded[1:])):
        raise ParseError("Excluded line ranges must be ordered and non-overlapping")
    document_metadata = dict(metadata or {})
    warnings: list[ParseWarning] = []
    inferred_title = path.stem
    if source_type == "pdf":
        sections, warnings = _pdf_sections(path)
    else:
        try:
            text = _clean_text(path.read_text(encoding="utf-8-sig"))
        except (UnicodeError, OSError) as error:
            raise ParseError("Text input must be readable UTF-8 (optional BOM)") from error
        if source_type == "markdown":
            sections, front_matter, first_heading = _markdown_sections(text, excluded)
            document_metadata["front_matter"] = front_matter
            front_title = front_matter.get("title")
            inferred_title = (
                front_title.strip() if isinstance(front_title, str) and front_title.strip()
                else first_heading or path.stem
            )
        else:
            lines = text.splitlines(keepends=True)
            section = _text_section(lines, 0, len(lines), [], [])
            sections = [section] if section is not None else []
    if not sections:
        raise ParseError("Document has no indexable text after parsing")
    return ParsedDocument(
        document_id=document_id,
        title=title if title is not None else inferred_title,
        source_path=source_path if source_path is not None else str(path),
        source_type=source_type,
        sections=sections,
        metadata=document_metadata,
        warnings=warnings,
    )
