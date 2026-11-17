"""Validate the inventory and select only explicitly prepared retrieval files."""

from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
import yaml

from app.ingestion.schemas import LineRange


class ManifestError(ValueError):
    """Safe, actionable inventory errors, without document body contents."""


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_type: Literal["markdown", "pdf", "txt", "csv"]
    path: str | None
    ingestion_status: Literal["prepared", "deferred_csv_strategy", "deferred_content_review", "review_required"]
    language: str | None = None
    category: str | None = None
    source_url: str | None = None
    organization: str | None = None
    published_at: date | str | None = None
    license: str = "review_required"
    visibility: str = "private"
    faq_line_ranges: list[tuple[int, int]] = Field(default_factory=list)

    @field_validator("source_path")
    @classmethod
    def canonical_source(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value != path.as_posix() or value == ".":
            raise ValueError("source_path must be a canonical relative path")
        return value

    def source_metadata(self) -> dict:
        return self.model_dump(include={
            "language", "category", "source_url", "organization", "published_at", "license", "visibility",
        })

    def excluded_ranges(self) -> list[LineRange]:
        return [LineRange(start=start, end=end) for start, end in self.faq_line_ranges]


class Manifest(BaseModel):
    schema_version: Literal[1]
    source_root: str
    path_base: Literal["manifest_directory"]
    documents: list[ManifestEntry] = Field(min_length=1)


def load_manifest(path: Path) -> Manifest:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = Manifest.model_validate(payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as error:
        raise ManifestError("Manifest is unreadable or invalid; check schema_version and required fields") from error
    for field in ("id", "source_path", "path"):
        values = [getattr(entry, field) for entry in manifest.documents if getattr(entry, field) is not None]
        if len(values) != len(set(values)):
            raise ManifestError(f"Manifest contains duplicate {field} values")
    if any(entry.ingestion_status == "prepared" and entry.path is None for entry in manifest.documents):
        raise ManifestError("Every prepared document must have a retrieval path")
    return manifest


def retrieval_path(root: Path, entry: ManifestEntry) -> Path:
    if entry.ingestion_status != "prepared" or entry.path is None:
        raise ManifestError(f"Document {entry.id} is not prepared for ingestion")
    relative = PurePosixPath(entry.path)
    if (
        relative.is_absolute() or ".." in relative.parts or not relative.parts
        or relative.parts[0] != "retrieval" or entry.path != relative.as_posix()
    ):
        raise ManifestError(f"Document {entry.id} must reference a relative path under retrieval/")
    candidate = (root / entry.path).resolve()
    if not candidate.is_relative_to(root.resolve() / "retrieval"):
        raise ManifestError(f"Document {entry.id} resolves outside retrieval/")
    expected = {"markdown": ".md", "pdf": ".pdf", "txt": ".txt"}.get(entry.source_type)
    if expected is None or candidate.suffix.lower() != expected or not candidate.is_file():
        raise ManifestError(f"Document {entry.id} has an unsupported, missing, or mismatched file")
    return candidate


def select_entries(
    manifest: Manifest, root: Path, *, document_ids: list[str], files: list[Path],
) -> list[ManifestEntry]:
    entries = manifest.documents
    if document_ids:
        known = {entry.id for entry in entries}
        if set(document_ids) - known:
            raise ManifestError("One or more requested document IDs are not in the Manifest")
        selected = [entry for entry in entries if entry.id in document_ids]
    elif files:
        requested = {(file if file.is_absolute() else root / file).resolve() for file in files}
        selected = [entry for entry in entries if entry.path is not None and (root / entry.path).resolve() in requested]
        if len(selected) != len(requested):
            raise ManifestError("--file must match a prepared Manifest path, not an original or evaluation file")
    else:
        selected = [entry for entry in entries if entry.ingestion_status == "prepared" and entry.path is not None]
    if not selected:
        raise ManifestError("No prepared documents selected")
    for entry in selected:
        retrieval_path(root, entry)
    return selected
