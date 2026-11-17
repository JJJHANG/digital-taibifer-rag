"""Atomically publish a complete version, serializing writers per source."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ingestion.embedding import EmbeddingBatch, validate_vectors
from app.ingestion.schemas import ChunkedDocument
from app.models import Document, DocumentChunk, DocumentVersion
from app.models.documents import EMBEDDING_DIMENSIONS


@dataclass(frozen=True)
class StoredIndex:
    status: Literal["indexed", "unchanged"]
    document_id: UUID
    version_id: UUID
    version_number: int
    chunks: int


def _matches(version: DocumentVersion, content_hash: str, fingerprint: str, model: str) -> bool:
    return (
        version.status == "ready" and version.content_hash == content_hash
        and version.embedding_model == model
        and version.chunking_config.get("index_fingerprint") == fingerprint
    )


class DocumentIndexRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def find_current(
        self, source_path: str, content_hash: str, fingerprint: str, model: str, chunks: int,
    ) -> StoredIndex | None:
        async with self._sessions() as session:
            row = (await session.execute(
                select(Document.id, DocumentVersion)
                .join(DocumentVersion, Document.active_version_id == DocumentVersion.id)
                .where(Document.source_path == source_path)
            )).one_or_none()
            if row is not None and _matches(row[1], content_hash, fingerprint, model):
                return StoredIndex("unchanged", row[0], row[1].id, row[1].version_number, chunks)
        return None

    async def save(
        self, document: ChunkedDocument, embeddings: EmbeddingBatch,
        *, content_hash: str, fingerprint: str,
    ) -> StoredIndex:
        validate_vectors(embeddings.vectors, count=len(document.chunks), dimensions=EMBEDDING_DIMENSIONS)
        if embeddings.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError("Embedding dimensions differ from the database schema")
        first = document.chunks[0].metadata
        async with self._sessions() as session, session.begin():
            await session.execute(insert(Document).values(
                title=first.title, source_path=first.source_path, source_type=first.source_type,
                source_url=first.document_metadata.get("source_url"),
            ).on_conflict_do_nothing(index_elements=[Document.source_path]))
            stored = (await session.scalars(
                select(Document).where(Document.source_path == first.source_path).with_for_update()
            )).one()
            if stored.active_version_id is not None:
                active = await session.get(DocumentVersion, stored.active_version_id)
                if active is not None and _matches(active, content_hash, fingerprint, embeddings.model):
                    return StoredIndex("unchanged", stored.id, active.id, active.version_number, len(document.chunks))
            number = (await session.scalar(
                select(func.coalesce(func.max(DocumentVersion.version_number), 0))
                .where(DocumentVersion.document_id == stored.id)
            )) + 1
            version = DocumentVersion(
                document_id=stored.id, content_hash=content_hash, version_number=number,
                status="ready", embedding_model=embeddings.model,
                chunking_config={**document.chunking_config.model_dump(mode="json"), "index_fingerprint": fingerprint},
            )
            session.add(version)
            await session.flush()
            warnings = [warning.model_dump(mode="json") for warning in document.warnings]
            session.add_all([
                DocumentChunk(
                    document_version_id=version.id, chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    chunk_metadata={**chunk.metadata.model_dump(mode="json"), "parse_warnings": warnings},
                    embedding=vector,
                )
                for chunk, vector in zip(document.chunks, embeddings.vectors, strict=True)
            ])
            await session.flush()
            stored.title, stored.source_type = first.title, first.source_type
            stored.source_url = first.document_metadata.get("source_url")
            stored.active_version_id = version.id
            result = StoredIndex("indexed", stored.id, version.id, version.version_number, len(document.chunks))
        return result
