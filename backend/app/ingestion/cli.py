"""Manifest-driven ingestion CLI; stdout is one JSON report, progress is stderr."""

import argparse
import asyncio
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

from openai import OpenAIError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import EmbeddingSettings, Settings
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedding import EmbeddingPlan, OpenAIEmbeddingProvider, plan_embedding_batches
from app.ingestion.indexer import EmbeddingIndexer
from app.ingestion.parsers import ParseError, parse_document
from app.ingestion.schemas import ChunkedDocument, ChunkingConfig
from app.knowledge.manifest import ManifestEntry, ManifestError, load_manifest, retrieval_path, select_entries
from app.knowledge.prepare import QUESTION, split_faq
from app.repositories.document_index import DocumentIndexRepository


@dataclass(frozen=True)
class PreparedInput:
    entry: ManifestEntry
    document: ChunkedDocument
    content_hash: str
    embedding_plan: EmbeddingPlan

    def preview(self) -> dict:
        return {
            "id": self.entry.id, "title": self.entry.title, "source_path": self.entry.source_path,
            "status": "planned", "chunks": len(self.document.chunks),
            "content_hash": self.content_hash,
            "estimated_tokens": sum(self.embedding_plan.token_counts),
            "estimated_requests": len(self.embedding_plan.batches),
            "warnings": [warning.model_dump(mode="json") for warning in self.document.warnings],
        }


def prepare_input(root: Path, entry: ManifestEntry, chunker: RecursiveChunker, settings: EmbeddingSettings) -> PreparedInput:
    path = retrieval_path(root, entry)
    before = path.read_bytes()
    if entry.source_type == "markdown":
        text = before.decode("utf-8-sig")
        if split_faq(text).faq or QUESTION.search(text):
            raise ManifestError(f"Prepared document {entry.id} still contains FAQ/Q&A; rerun knowledge separation")
    parsed = parse_document(
        path, document_id=entry.id, title=entry.title, source_path=entry.source_path,
        metadata=entry.source_metadata(), excluded_line_ranges=entry.excluded_ranges(),
    )
    chunked = chunker.split_document(parsed)
    if path.read_bytes() != before:
        raise ManifestError(f"Document {entry.id} changed during preparation; retry with a stable snapshot")
    plan = plan_embedding_batches([chunk.content for chunk in chunked.chunks], batch_size=settings.embedding_batch_size)
    return PreparedInput(entry, chunked, sha256(before).hexdigest(), plan)


def safe_error(error: Exception) -> dict:
    """Never serialize API bodies, SQL params, document text, or credentials."""
    result = {"type": type(error).__name__}
    if isinstance(error, ManifestError):
        result["message"] = str(error)
    elif isinstance(error, OpenAIError):
        result["message"] = "OpenAI request failed; check credentials, quota, connectivity and model limits"
        result["http_status"] = getattr(error, "status_code", None)
    elif isinstance(error, SQLAlchemyError):
        result["message"] = "Database operation failed; check connection, migrations and constraints"
    else:
        result["message"] = "Preparation or indexing failed; check file encoding, structure, configuration and input consistency"
    return result


async def ingest_prepared(
    prepared: list[PreparedInput], report: dict, *, continue_on_error: bool,
) -> int:
    engine = create_async_engine(Settings().database_url, pool_pre_ping=True)
    try:
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with OpenAIEmbeddingProvider(EmbeddingSettings()) as provider:
            indexer = EmbeddingIndexer(provider, DocumentIndexRepository(sessions))
            for index, item in enumerate(prepared):
                row = report["documents"][index]
                try:
                    result = await indexer.index(item.document, content_hash=item.content_hash)
                    stored = result.stored
                    row.update({
                        "status": stored.status, "document_id": str(stored.document_id),
                        "version_id": str(stored.version_id), "version_number": stored.version_number,
                        "usage": result.usage.model_dump(),
                    })
                except (OpenAIError, SQLAlchemyError, ValueError, OSError) as error:
                    row.update(status="failed", error=safe_error(error))
                    if not continue_on_error:
                        for remaining in report["documents"][index + 1:]:
                            remaining["status"] = "not_attempted"
                        break
                finally:
                    print(f"[{index + 1}/{len(prepared)}] {item.entry.id}: {row['status']}", file=sys.stderr, flush=True)
    finally:
        await engine.dispose()
    failed = any(row["status"] in {"failed", "not_attempted"} for row in report["documents"])
    report["status"] = "failed" if failed else "completed"
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ingest.py", description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/knowledge/manifest.yaml"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--document-id", action="append", default=[], help="Manifest ID; repeat to select several")
    selection.add_argument("--file", type=Path, action="append", default=[], help="Prepared file, relative to Manifest or absolute")
    parser.add_argument("--dry-run", action="store_true", help="Parse/chunk/count tokens without API calls or DB connections")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--continue-on-error", action="store_true", help="Try remaining documents after an indexing failure")
    arguments = parser.parse_args(argv)
    report: dict = {"status": "failed", "documents": []}
    try:
        settings = EmbeddingSettings()
        config = ChunkingConfig(chunk_size=arguments.chunk_size, chunk_overlap=arguments.chunk_overlap)
        manifest_path = arguments.manifest.resolve()
        manifest = load_manifest(manifest_path)
        entries = select_entries(manifest, manifest_path.parent, document_ids=arguments.document_id, files=arguments.file)
        chunker = RecursiveChunker(config)
        # Complete every file's preflight before opening the SDK or DB engine.
        prepared: list[PreparedInput] = []
        for entry in entries:
            try:
                prepared.append(prepare_input(manifest_path.parent, entry, chunker, settings))
            except ManifestError:
                raise
            except (ValueError, OSError) as error:
                raise ManifestError(
                    f"Preparation failed for {entry.id} ({type(error).__name__}); "
                    "check encoding, structure, source ranges and token limits"
                ) from error
        selected_ids = {entry.id for entry in entries}
        report.update({
            "status": "dry_run" if arguments.dry_run else "running",
            "model": settings.embedding_model, "dimensions": settings.embedding_dimensions,
            "chunking_config": config.model_dump(mode="json"),
            "documents": [item.preview() for item in prepared],
            "excluded": [{"id": entry.id, "source_type": entry.source_type,
                          "reason": "not_selected" if entry.ingestion_status == "prepared" else entry.ingestion_status}
                         for entry in manifest.documents if entry.id not in selected_ids],
            "totals": {
                "selected_documents": len(prepared),
                "planned_chunks": sum(len(item.document.chunks) for item in prepared),
                "estimated_tokens_before_dedup": sum(sum(item.embedding_plan.token_counts) for item in prepared),
                "estimated_requests_before_dedup": sum(len(item.embedding_plan.batches) for item in prepared),
            },
        })
        exit_code = 0 if arguments.dry_run else asyncio.run(ingest_prepared(
            prepared, report, continue_on_error=arguments.continue_on_error,
        ))
    except (ManifestError, ParseError, ValidationError, ValueError, OSError, OpenAIError, SQLAlchemyError) as error:
        report.update(status="failed", error=safe_error(error))
        for row in report["documents"]:
            if row["status"] == "planned":
                row["status"] = "not_attempted"
        exit_code = 2 if not report["documents"] else 1
    except KeyboardInterrupt:
        report.update(status="interrupted")
        for row in report["documents"]:
            if row["status"] == "planned":
                row["status"] = "not_attempted"
        exit_code = 130
    report.setdefault("totals", {}).update({
        "indexed": sum(row["status"] == "indexed" for row in report["documents"]),
        "unchanged": sum(row["status"] == "unchanged" for row in report["documents"]),
        "failed": sum(row["status"] == "failed" for row in report["documents"]),
        "usage": {key: sum(row.get("usage", {}).get(key, 0) for row in report["documents"])
                  for key in ("prompt_tokens", "total_tokens", "requests")},
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
