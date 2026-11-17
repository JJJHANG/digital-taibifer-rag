"""Synthetic fixtures only: private knowledge must never enter test sources."""

import pytest
import yaml

from app.knowledge.prepare import prepare_knowledge, split_faq


def test_faq_descendants_excluded_but_following_resources_preserved():
    source = "# Guide\nBody\n# FAQ 問答集\n**Q: Question?**\nA: Answer\n## Detail\nSecret\n# Resources\nLink\n"
    result = split_faq(source)
    assert result.retrieval == "# Guide\nBody\n# Resources\nLink\n"
    assert result.faq == "# FAQ 問答集\n**Q: Question?**\nA: Answer\n## Detail\nSecret\n"
    assert result.faq_line_ranges == [(3, 7)]


def test_heading_inside_front_matter_or_code_is_not_a_section():
    source = "---\n# FAQ 問答集\ntitle: example\n---\n```md\n# FAQ 問答集\n```\n# Guide\nContent\n"
    assert split_faq(source).retrieval == source
    assert split_faq(source).faq == ""


def test_multiple_faq_sections_and_no_faq():
    source = "# Intro\nA\n## FAQ\nQ1\n## Next\nB\n# FAQ 問答集\nQ2\n"
    result = split_faq(source)
    assert result.retrieval == "# Intro\nA\n## Next\nB\n"
    assert result.faq_line_ranges == [(3, 4), (7, 8)]
    assert split_faq("# Guide\nBody\n").retrieval == "# Guide\nBody\n"


@pytest.mark.parametrize("source", ["---\ntitle: unclosed\n", "```md\n# FAQ\n"])
def test_unclosed_structures_fail_instead_of_leaking_faq(source):
    with pytest.raises(ValueError):
        split_faq(source)


def test_private_snapshot_manifest_and_originals(tmp_path):
    source = tmp_path / "originals"
    source.mkdir()
    markdown = "# Guide\nEvidence\n# FAQ 問答集\n**Q: Example?**\nA: Example answer\n"
    (source / "guide.md").write_text(markdown, encoding="utf-8")
    (source / "terms.csv").write_text('term,definition\na,"multi\nline"\n', encoding="utf-8")
    output = tmp_path / "private" / "snapshot"
    report = prepare_knowledge(source, output)
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    entry, csv_entry = manifest["documents"]
    assert (source / "guide.md").read_text() == markdown
    assert (output / entry["path"]).read_text() == "# Guide\nEvidence\n"
    assert "Example answer" in (output / entry["evaluation_path"]).read_text()
    assert entry["evaluation_status"] == "needs_human_review"
    assert entry["license"] == "review_required"
    assert csv_entry["path"] is None
    assert csv_entry["csv_data_rows"] == 1
    assert report["faq_question_markers"] == 1
    assert (output / ".gitignore").read_text() == "*\n"
    with pytest.raises(FileExistsError):
        prepare_knowledge(source, output)
    other = tmp_path / "private" / "second"
    prepare_knowledge(source, other)
    assert yaml.safe_load((other / "manifest.yaml").read_text())["documents"][0]["id"] == entry["id"]


def test_unclassified_questions_abort_snapshot(tmp_path):
    source = tmp_path / "originals"
    source.mkdir()
    (source / "guide.md").write_text("# Guide\n**Q: Unclassified?**\nA: Answer\n")
    output = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="Q&A outside"):
        prepare_knowledge(source, output)
    assert not output.exists()


def test_output_cannot_modify_original_directory(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        prepare_knowledge(tmp_path, tmp_path / "snapshot")
