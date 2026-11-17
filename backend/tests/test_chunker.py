"""Verify boundaries, content coverage, and exact provenance on synthetic text."""

import pytest
from pydantic import ValidationError

from app.ingestion.chunker import RecursiveChunker
from app.ingestion.parsers import parse_document
from app.ingestion.schemas import (
    ChunkedDocument, ChunkingConfig, LineRange, ParsedDocument, ParsedSection, ParseWarning,
)


def document_for(text, **section_fields):
    return ParsedDocument(
        document_id="guide", title="測試指南", source_path="original.md", source_type="markdown",
        metadata={"language": "zh-TW", "nested": {"category": "example"}},
        sections=[ParsedSection(content=text, **section_fields)],
    )


def assert_coverage(document, result):
    for section_index, section in enumerate(document.sections):
        covered = set()
        for chunk in result.chunks:
            metadata = chunk.metadata
            if metadata.section_index != section_index:
                continue
            assert section.content[metadata.char_start:metadata.char_end] == chunk.content
            covered.update(range(metadata.char_start, metadata.char_end))
        assert all(index in covered for index, char in enumerate(section.content) if not char.isspace())


def test_character_fallback_has_bounded_size_and_exact_overlap():
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    document = document_for(text)
    result = RecursiveChunker(ChunkingConfig(chunk_size=10, chunk_overlap=3)).split_document(document)
    assert [chunk.metadata.char_start for chunk in result.chunks] == [0, 7, 14, 21, 28]
    assert all(len(chunk.content) <= 10 for chunk in result.chunks)
    assert result.chunks[0].content[-3:] == result.chunks[1].content[:3]
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(5))
    assert_coverage(document, result)


@pytest.mark.parametrize("text", [
    "同一段落。\n\n" * 60,
    "生物多樣性與開放資料。" * 80,
    "first\n\nsecond\n\nthird\n\n" * 30,
    "\n" * 50 + "content" + " " * 50,
])
def test_repeated_multilingual_and_whitespace_content_keeps_coverage(text):
    document = document_for(text)
    config = ChunkingConfig(chunk_size=35, chunk_overlap=8)
    result = RecursiveChunker(config).split_document(document)
    assert all(0 < len(chunk.content) <= 35 for chunk in result.chunks)
    assert_coverage(document, result)


def test_zero_overlap_keeps_separators_and_short_documents_are_not_duplicated():
    document = document_for("First sentence。Second sentence！Third sentence？")
    chunker = RecursiveChunker(ChunkingConfig(chunk_size=20, chunk_overlap=0))
    result = chunker.split_document(document)
    assert "".join(chunk.content for chunk in result.chunks) == document.sections[0].content
    short = chunker.split_document(document_for("Short text"))
    assert len(short.chunks) == 1


def test_sections_and_pages_are_never_merged_and_warnings_survive():
    document = ParsedDocument(
        document_id="pdf", title="Guide", source_path="guide.pdf", source_type="pdf",
        sections=[
            ParsedSection(content="Page one", page_number=1),
            ParsedSection(content="Page three", page_number=3),
        ],
        warnings=[ParseWarning(code="page_without_text", page_number=2, message="Review")],
    )
    result = RecursiveChunker().split_document(document)
    assert [chunk.metadata.page_number for chunk in result.chunks] == [1, 3]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]
    assert result.warnings == document.warnings
    assert all(chunk.metadata.line_range is None for chunk in result.chunks)
    assert all(chunk.metadata.source_line_ranges == [] for chunk in result.chunks)


def test_chunk_line_ranges_are_clipped_across_removed_faq(tmp_path):
    path = tmp_path / "prepared.md"
    path.write_text("# Main\nAAA\nBBB\nCCC\n")
    document = parse_document(
        path, document_id="guide", source_path="original.md",
        excluded_line_ranges=[LineRange(start=3, end=5)],
    )
    result = RecursiveChunker(ChunkingConfig(chunk_size=9, chunk_overlap=0)).split_document(document)
    assert result.chunks[0].metadata.line_range == LineRange(start=1, end=1)
    assert result.chunks[1].metadata.line_range == LineRange(start=2, end=3)
    assert result.chunks[1].metadata.source_line_ranges == [
        LineRange(start=2, end=2), LineRange(start=6, end=6)
    ]
    assert all(chunk.metadata.heading_path == ["Main"] for chunk in result.chunks)
    assert_coverage(document, result)


def test_mid_line_character_splits_keep_the_same_line_number():
    document = document_for(
        "abcdefghijklmno", line_range=LineRange(start=8, end=8),
        source_line_ranges=[LineRange(start=15, end=15)],
    )
    result = RecursiveChunker(ChunkingConfig(chunk_size=6, chunk_overlap=2)).split_document(document)
    assert all(chunk.metadata.line_range == LineRange(start=8, end=8) for chunk in result.chunks)
    assert all(chunk.metadata.source_line_ranges == [LineRange(start=15, end=15)] for chunk in result.chunks)


def test_metadata_is_namespaced_independent_and_json_serializable():
    document = document_for("A" * 40, heading_path=["Main", "Sub"])
    document.metadata["char_start"] = "must not overwrite calculated coordinates"
    chunker = RecursiveChunker(ChunkingConfig(chunk_size=15, chunk_overlap=2))
    result = chunker.split_document(document)
    assert result == chunker.split_document(document)
    assert result.chunks[0].metadata.char_start == 0
    result.chunks[0].metadata.document_metadata["nested"]["category"] = "changed"
    assert document.metadata["nested"]["category"] == "example"
    assert result.chunks[1].metadata.document_metadata["nested"]["category"] == "example"
    assert ChunkedDocument.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("settings", [
    {"chunk_size": 0}, {"chunk_overlap": -1}, {"chunk_size": 5, "chunk_overlap": 5},
    {"chunk_size": 5, "chunk_overlap": 6}, {"chunk_size": 8.5}, {"separators": ("\n",)},
])
def test_invalid_config_is_rejected(settings):
    with pytest.raises(ValidationError):
        ChunkingConfig(**settings)


def test_invalid_source_coordinates_and_blank_input_are_rejected():
    with pytest.raises(ValueError, match="line_range"):
        RecursiveChunker().split_document(document_for("One line", line_range=LineRange(start=1, end=3)))
    with pytest.raises(ValueError, match="non-whitespace"):
        RecursiveChunker().split_document(document_for(" \n\t"))
