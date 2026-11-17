"""Prepare private retrieval documents and held-out FAQ material.

Only Markdown FAQ sections are separated here. PDF/CSV are inventoried,
never parsed or silently enrolled for ingestion. Original files are read-only.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from uuid import NAMESPACE_URL, uuid5

import yaml


HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
FAQ_TITLE = re.compile(r"(?:FAQ(?:\s*問答集)?|常見問題|問答集)", re.IGNORECASE)
QUESTION = re.compile(r"^\s*\*\*Q\s*[:：]", re.MULTILINE)
SOURCE_TYPES = {".md": "markdown", ".csv": "csv", ".pdf": "pdf", ".txt": "txt"}


@dataclass(frozen=True)
class MarkdownSplit:
    retrieval: str
    faq: str
    faq_line_ranges: list[tuple[int, int]]


def split_faq(markdown: str) -> MarkdownSplit:
    """Exclude FAQ headings and descendants, preserving following peer sections.

    This is a preparation rule for ATX headings, not the S1-04 Markdown parser.
    Code fences and leading YAML front matter cannot start FAQ sections.
    Line ranges are inclusive, one-based positions in the original document.
    """
    lines = markdown.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    front_matter = bool(lines and lines[0].strip() == "---")
    fence_char = ""
    fence_length = 0
    for index, line in enumerate(lines):
        if front_matter:
            if index > 0 and line.strip() in {"---", "..."}:
                front_matter = False
            continue
        fence = FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length and not line[fence.end():].strip():
                fence_char, fence_length = "", 0
            continue
        if not fence_char and (match := HEADING.match(line)):
            headings.append((index, len(match.group(1)), match.group(2)))

    if front_matter or fence_char:
        raise ValueError("Unclosed front matter or code fence; inspect Markdown before preparing")

    ranges: list[tuple[int, int]] = []
    for position, (start, level, title) in enumerate(headings):
        if not FAQ_TITLE.fullmatch(title.strip()):
            continue
        if ranges and start < ranges[-1][1]:
            continue
        end = next(
            (offset for offset, depth, _ in headings[position + 1:] if depth <= level),
            len(lines),
        )
        ranges.append((start, end))

    retrieval: list[str] = []
    faq: list[str] = []
    cursor = 0
    for start, end in ranges:
        retrieval.extend(lines[cursor:start])
        faq.extend(lines[start:end])
        cursor = end
    retrieval.extend(lines[cursor:])
    return MarkdownSplit(
        retrieval="".join(retrieval),
        faq="".join(faq),
        faq_line_ranges=[(start + 1, end) for start, end in ranges],
    )


def prepare_knowledge(source: Path, output: Path) -> dict:
    """Build a fresh private snapshot, publishing it only after success.

    Existing output is never overwritten, preserving later human annotations.
    IDs depend on relative source paths, not document content. Renaming a source
    therefore changes its generated ID and needs a deliberate manifest update.
    """
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError("Source must be an existing directory")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the original source directory")
    if output.exists():
        raise FileExistsError("Output already exists; use a new snapshot directory")

    output.parent.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []
    ignored: list[str] = []
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        (staging / "retrieval").mkdir(parents=True)
        (staging / "evaluation" / "faq").mkdir(parents=True)
        (staging / ".gitignore").write_text("*\n", encoding="utf-8")
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            if path.is_symlink():
                raise ValueError(f"Source symlinks need explicit review: {relative}")
            source_type = SOURCE_TYPES.get(path.suffix.lower())
            if source_type is None or any(part.startswith(".") for part in Path(relative).parts):
                ignored.append(relative)
                continue
            document_id = "doc-" + uuid5(NAMESPACE_URL, "digital-taibifer/" + relative).hex
            entry = {
                "id": document_id,
                "source_path": relative,
                "title": path.stem,
                "source_type": source_type,
                "path": None,
                "language": None,
                "category": None,
                "source_url": None,
                "organization": None,
                "published_at": None,
                "license": "review_required",
                "visibility": "private",
                "ingestion_status": "review_required",
                "evaluation_path": None,
                "evaluation_status": "not_prepared",
                "faq_line_ranges": [],
                "faq_question_markers": 0,
            }
            if source_type == "markdown":
                split = split_faq(path.read_text(encoding="utf-8-sig"))
                if QUESTION.search(split.retrieval):
                    raise ValueError(f"Q&A outside a recognized FAQ section: {relative}")
                entry["path"] = f"retrieval/{document_id}.md"
                (staging / entry["path"]).write_text(split.retrieval, encoding="utf-8")
                entry["ingestion_status"] = "prepared"
                if split.faq:
                    entry["evaluation_path"] = f"evaluation/faq/{document_id}.md"
                    (staging / entry["evaluation_path"]).write_text(split.faq, encoding="utf-8")
                    entry["evaluation_status"] = "needs_human_review"
                    entry["faq_line_ranges"] = [list(bounds) for bounds in split.faq_line_ranges]
                    entry["faq_question_markers"] = len(QUESTION.findall(split.faq))
            elif source_type == "csv":
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    entry["csv_columns"] = next(reader, [])
                    entry["csv_data_rows"] = sum(1 for _ in reader)
                entry["ingestion_status"] = "deferred_csv_strategy"
            else:
                # PDFs may duplicate FAQ answers or contain other held-out text.
                entry["ingestion_status"] = "deferred_content_review"
            documents.append(entry)

        if not documents:
            raise ValueError("No supported source files found")
        manifest = {
            "schema_version": 1,
            "source_root": str(source),
            "path_base": "manifest_directory",
            "retrieval_policy": "Only prepared entries with a non-null path; never source_root or evaluation/",
            "documents": documents,
            "ignored_files": ignored,
        }
        (staging / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        report = {
            "documents": len(documents),
            "prepared_markdown": sum(entry["ingestion_status"] == "prepared" for entry in documents),
            "faq_documents": sum(entry["evaluation_path"] is not None for entry in documents),
            "faq_question_markers": sum(entry["faq_question_markers"] for entry in documents),
            "deferred_documents": sum(entry["path"] is None for entry in documents),
        }
        (staging / "inventory.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = prepare_knowledge(arguments.source, arguments.output)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
