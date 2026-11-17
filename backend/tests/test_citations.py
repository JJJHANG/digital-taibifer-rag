"""Citation identity, context selection, prose syntax and backend provenance."""

import json
from uuid import uuid4

import pytest

from app.rag.citations import build_citation_map, CitationContextError, validate_citations
from app.rag.prompt import build_rag_prompt
from app.rag.schemas import CitationReport, PromptConfig, RetrievedChunk


def chunk(content="Synthetic evidence", **metadata):
    return RetrievedChunk(
        chunk_id=uuid4(), document_id=uuid4(), document_version_id=uuid4(), version_number=2,
        chunk_index=3, title="Backend title", source_path="original.md", source_type="markdown",
        content=content, distance=0.2, score=0.8,
        metadata={"heading_path": ["Topic"], "page_number": None,
                  "source_line_ranges": [{"start": 8, "end": 9}, {"start": 15, "end": 16}],
                  "document_metadata": {"source_url": "https://example.org/guide"}, **metadata},
    )


def test_valid_markers_map_to_backend_sources_in_first_appearance_order():
    chunks = [chunk(), chunk("Second evidence")]
    prompt = build_rag_prompt("question", chunks)
    report = validate_citations("Fact [S2][S1]. Repeated [S2].", build_citation_map(prompt, chunks))
    assert report.status == "valid" and report.cited_ids == ["S2", "S1"]
    assert [source.chunk_id for source in report.sources] == [chunks[1].chunk_id, chunks[0].chunk_id]
    assert report.sources[0].document_version_id == chunks[1].document_version_id
    assert report.sources[0].title == "Backend title"
    assert str(report.sources[0].source_url) == "https://example.org/guide"
    assert report.sources[0].source_line_ranges[1].start == 15
    assert report.grounding_checked is False
    assert CitationReport.model_validate_json(report.model_dump_json()) == report


def test_mapping_uses_prompt_selected_order_not_retrieval_ranking():
    big, small = chunk("long evidence " * 3000), chunk("Short evidence")
    prompt = build_rag_prompt("question", [big, small], config=PromptConfig(max_context_tokens=300))
    sources = build_citation_map(prompt, [big, small])
    assert list(sources) == ["S1"] and sources["S1"].chunk_id == small.chunk_id
    report = validate_citations("Invented [S2]", sources)
    assert report.status == "invalid" and report.sources == []


@pytest.mark.parametrize("marker", ["[S999]", "[S0]", "[S01]", "[s1]", "[S 1]", "[ S1]", "[S1,S2]", "[Sabc]", "[S１]"])
def test_unknown_or_malformed_markers_are_not_silently_fixed(marker):
    source = chunk()
    sources = build_citation_map(build_rag_prompt("question", [source]), [source])
    report = validate_citations(f"Valid [S1], then invalid {marker}", sources)
    assert report.status == "invalid" and marker in report.invalid_markers
    assert report.sources == []


def test_code_literals_are_not_treated_as_prose_citations():
    source = chunk()
    sources = build_citation_map(build_rag_prompt("question", [source]), [source])
    report = validate_citations("`[S999]` is an example.\n\n```text\n[S888]\n```\n\nActual fact **[S1]**.", sources)
    assert report.status == "valid" and report.cited_ids == ["S1"]


def test_model_link_is_not_an_authoritative_citation_url():
    source = chunk()
    sources = build_citation_map(build_rag_prompt("question", [source]), [source])
    report = validate_citations("Fact [S1](https://invented.example).", sources)
    assert report.status == "invalid" and report.sources == []


def test_missing_and_not_applicable_are_explicit():
    assert validate_citations("Uncited answer", {}).status == "missing"
    assert validate_citations("Explicit refusal", {}, expected=False).status == "not_applicable"


@pytest.mark.parametrize("fault", ["label", "content", "selected_id", "duplicate_selected", "conflicting_identity"])
def test_tampered_context_cannot_create_a_trusted_source_map(fault):
    source = chunk()
    chunks = [source]
    prompt = build_rag_prompt("question", chunks)
    payload = json.loads(prompt.messages[1].content)
    if fault == "label":
        payload["sources"][0]["label"] = "S999"
    elif fault == "content":
        payload["sources"][0]["content"] = "Different evidence"
    elif fault == "selected_id":
        prompt.selected_chunk_ids = [uuid4()]
    elif fault == "duplicate_selected":
        prompt.selected_chunk_ids.append(source.chunk_id)
    else:
        other = source.model_copy(deep=True)
        other.content = "Conflicting body for the same ID"
        chunks.append(other)
    prompt.messages[1].content = json.dumps(payload)
    with pytest.raises(CitationContextError):
        build_citation_map(prompt, chunks)


def test_invalid_backend_location_or_url_is_rejected():
    source = chunk(page_number=-1)
    with pytest.raises(CitationContextError, match="metadata"):
        build_citation_map(build_rag_prompt("question", [source]), [source])
    source = chunk(document_metadata={"source_url": "javascript:alert(1)"})
    with pytest.raises(CitationContextError, match="metadata"):
        build_citation_map(build_rag_prompt("question", [source]), [source])
