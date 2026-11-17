"""Synthetic inputs only; test structure, provenance, and failure behavior."""

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.ingestion.parsers import ParseError, parse_document
from app.ingestion.schemas import LineRange, ParsedDocument
from app.knowledge.prepare import split_faq


def write_pdf(path, pages, *, password=None):
    writer = PdfWriter()
    for content in pages:
        page = writer.add_blank_page(width=300, height=300)
        if content is None:
            continue
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
        })
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 200 Td ({content}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    if password:
        writer.encrypt(password)
    writer.write(path)


def test_markdown_hierarchy_preserves_tables_and_fenced_code(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text(
        "---\ntitle: 文件標題\ncategory: example\n---\n\n"
        "# 主題\n導言\n\n### 子題\n| 欄位 | 說明 |\n| --- | --- |\n| a | b |\n"
        "\n```python\n# 這不是章節\nprint('hello')\n```\n\n## 另一節\n結語\n",
        encoding="utf-8",
    )
    metadata = {"language": "zh-TW", "nested": {"key": "value"}}
    document = parse_document(path, document_id="guide", metadata=metadata)
    assert document.title == "文件標題"
    assert [section.heading_path for section in document.sections] == [
        ["主題"], ["主題", "子題"], ["主題", "另一節"]
    ]
    assert "| a | b |" in document.sections[1].content
    assert "```python\n# 這不是章節\nprint('hello')\n```" in document.sections[1].content
    assert document.sections[0].line_range == LineRange(start=6, end=7)
    assert "title:" not in document.sections[0].content
    assert document.metadata["front_matter"]["category"] == "example"
    assert "front_matter" not in metadata
    assert ParsedDocument.model_validate_json(document.model_dump_json()) == document


def test_setext_heading_and_quoted_headings(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("前言\n\nMain\n====\nText\n\n> # Quoted heading\n\nSub\n---\nEnd\n")
    document = parse_document(path, document_id="guide", title="Manifest title")
    assert document.title == "Manifest title"
    assert [section.heading_path for section in document.sections] == [[], ["Main"], ["Main", "Sub"]]
    assert "> # Quoted heading" in document.sections[1].content


def test_prepared_markdown_retains_original_source_coordinates(tmp_path):
    original = "# Main\nEvidence\n# FAQ 問答集\n**Q: Hidden?**\nA: Hidden answer\n# Resources\nLink\n"
    split = split_faq(original)
    path = tmp_path / "prepared.md"
    path.write_text(split.retrieval)
    document = parse_document(
        path, document_id="original", source_path="raw/original.md",
        excluded_line_ranges=[LineRange(start=start, end=end) for start, end in split.faq_line_ranges],
    )
    assert document.source_path == "raw/original.md"
    resources = document.sections[1]
    assert resources.line_range == LineRange(start=3, end=4)
    assert resources.source_line_ranges == [LineRange(start=6, end=7)]
    assert "Hidden answer" not in "\n".join(section.content for section in document.sections)


def test_disjoint_original_ranges_within_one_section(tmp_path):
    path = tmp_path / "prepared.md"
    path.write_text("# Main\nA\nB\nC\n")
    result = parse_document(
        path, document_id="guide", excluded_line_ranges=[LineRange(start=3, end=5)]
    )
    assert result.sections[0].source_line_ranges == [
        LineRange(start=1, end=2), LineRange(start=6, end=7)
    ]


def test_txt_utf8_bom_newlines_nulls_and_indentation(tmp_path):
    path = tmp_path / "text.TXT"
    path.write_bytes(b"\xef\xbb\xbf" + "臺灣\r\n\r\n  keep indentation\x00\rLast\n".encode())
    result = parse_document(path, document_id="text")
    assert result.source_type == "txt"
    assert result.sections[0].content == "臺灣\n\n  keep indentation\nLast"
    assert result.sections[0].line_range == LineRange(start=1, end=4)


@pytest.mark.parametrize("filename,content", [
    ("empty.txt", " \n\t"),
    ("empty.md", "---\ntitle: only metadata\n---\n"),
    ("unclosed.md", "---\ntitle: no closing delimiter\n"),
    ("bad_yaml.md", "---\ntitle: [\n---\nBody\n"),
    ("list_yaml.md", "---\n- invalid\n---\nBody\n"),
])
def test_invalid_or_empty_text_fails_explicitly(tmp_path, filename, content):
    path = tmp_path / filename
    path.write_text(content)
    with pytest.raises(ParseError):
        parse_document(path, document_id="bad")


def test_non_utf8_and_unsupported_csv_are_rejected(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(ParseError, match="UTF-8"):
        parse_document(path, document_id="legacy")
    with pytest.raises(ParseError, match="Unsupported"):
        parse_document(tmp_path / "terms.csv", document_id="csv")
    with pytest.raises(ParseError):
        parse_document(tmp_path / "missing.md", document_id="missing")


def test_line_range_validation(tmp_path):
    with pytest.raises(ValidationError):
        LineRange(start=4, end=2)
    path = tmp_path / "guide.md"
    path.write_text("# Guide\nBody\n")
    with pytest.raises(ParseError, match="ordered"):
        parse_document(path, document_id="guide", excluded_line_ranges=[
            LineRange(start=5, end=8), LineRange(start=7, end=10)
        ])


def test_pdf_physical_page_numbers_and_missing_text_warnings(tmp_path):
    path = tmp_path / "pages.pdf"
    write_pdf(path, ["First page", None, "Third page"])
    result = parse_document(path, document_id="pdf", metadata={"category": "example"})
    assert [section.page_number for section in result.sections] == [1, 3]
    assert result.sections[0].content == "First page"
    assert result.warnings[0].page_number == 2
    assert result.warnings[0].code == "page_without_text"
    assert result.metadata["category"] == "example"
    assert all(section.line_range is None for section in result.sections)


@pytest.mark.parametrize("kind", ["blank", "encrypted", "corrupt"])
def test_unreadable_pdfs_are_not_silent_successes(tmp_path, kind):
    path = tmp_path / "bad.pdf"
    if kind == "corrupt":
        path.write_bytes(b"not a PDF")
    else:
        write_pdf(path, [None] if kind == "blank" else ["Private"], password="secret" if kind == "encrypted" else None)
    with pytest.raises(ParseError):
        parse_document(path, document_id="bad")
