"""Real committed transactions in a disposable PostgreSQL database.

FakeProvider exists only in tests and never generates production embeddings.
"""

import asyncio
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedding import EmbeddingBatch, EmbeddingUsage
from app.ingestion.indexer import EmbeddingIndexer
from app.ingestion.schemas import ParsedDocument, ParsedSection
from app.models import Document, DocumentChunk, DocumentVersion
from app.repositories.document_index import DocumentIndexRepository


class FakeProvider:
    model = "text-embedding-3-small"
    dimensions = 1536

    def __init__(self):
        self.calls = 0
        self.fault = None
        self.barrier = None

    async def embed_documents(self, texts):
        self.calls += 1
        if self.barrier is not None:
            if self.calls >= 2:
                self.barrier.set()
            await self.barrier.wait()
        if self.fault == "api":
            raise RuntimeError("Synthetic API failure")
        vectors = [[1.0] + [0.0] * 1535 for _ in texts]
        if self.fault == "count":
            vectors.pop()
        elif self.fault == "nan":
            vectors[0][0] = float("nan")
        return EmbeddingBatch(model=self.model, dimensions=self.dimensions, vectors=vectors,
                              usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2, requests=1))


def sample_document(source_path=None, content="Synthetic sample knowledge"):
    return RecursiveChunker().split_document(ParsedDocument(
        document_id="synthetic-guide", title="Synthetic guide", source_path=source_path or f"test/{uuid4()}.md",
        source_type="markdown", metadata={"visibility": "private", "source_url": None},
        sections=[ParsedSection(content=content, heading_path=["Example"])],
    ))


def digest(text):
    return sha256(text.encode()).hexdigest()


async def setup(url):
    engine = create_async_engine(url)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    provider = FakeProvider()
    indexer = EmbeddingIndexer(provider, DocumentIndexRepository(sessions))
    return engine, sessions, provider, indexer


def test_persist_vectors_metadata_and_skip_identical_input(index_database_url):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            document = sample_document()
            first = await indexer.index(document, content_hash=digest("source bytes"))
            second = await indexer.index(document, content_hash=digest("source bytes"))
            assert first.stored.status == "indexed" and second.stored.status == "unchanged"
            assert first.stored.version_id == second.stored.version_id
            assert provider.calls == 1 and second.usage.requests == 0
            async with sessions() as session:
                stored = await session.get(Document, first.stored.document_id)
                assert stored.active_version_id == first.stored.version_id
                chunk = (await session.scalars(select(DocumentChunk).where(
                    DocumentChunk.document_version_id == first.stored.version_id
                ))).one()
                assert len(chunk.embedding) == 1536
                assert chunk.chunk_metadata["heading_path"] == ["Example"]
                assert chunk.chunk_metadata["document_metadata"]["visibility"] == "private"
                assert await session.scalar(select(DocumentChunk.embedding.cosine_distance([1.0] + [0.0] * 1535)).where(DocumentChunk.id == chunk.id)) == pytest.approx(0)
        finally:
            await engine.dispose()
    asyncio.run(scenario())


def test_metadata_changes_create_a_new_complete_version(index_database_url):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            document = sample_document()
            first = await indexer.index(document, content_hash=digest("same source"))
            document.chunks[0].metadata.title = "Updated metadata title"
            second = await indexer.index(document, content_hash=digest("same source"))
            assert second.stored.version_number == 2 and provider.calls == 2
            async with sessions() as session:
                stored = await session.get(Document, first.stored.document_id)
                assert stored.active_version_id == second.stored.version_id
                assert (await session.get(DocumentVersion, first.stored.version_id)).status == "ready"
        finally:
            await engine.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["api", "count", "nan"])
def test_failed_embedding_preserves_previous_active_version(index_database_url, fault):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            document = sample_document()
            first = await indexer.index(document, content_hash=digest("old source"))
            provider.fault = fault
            with pytest.raises((RuntimeError, ValueError)):
                await indexer.index(document, content_hash=digest("new source"))
            async with sessions() as session:
                stored = await session.get(Document, first.stored.document_id)
                assert stored.active_version_id == first.stored.version_id
                assert await session.scalar(select(func.count()).select_from(DocumentVersion).where(
                    DocumentVersion.document_id == stored.id
                )) == 1
        finally:
            await engine.dispose()
    asyncio.run(scenario())


def test_concurrent_same_source_publishes_only_one_version(index_database_url):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            provider.barrier = asyncio.Event()
            document = sample_document()
            first, second = await asyncio.wait_for(asyncio.gather(
                indexer.index(document, content_hash=digest("same source")),
                indexer.index(document, content_hash=digest("same source")),
            ), timeout=15)
            assert {first.stored.status, second.stored.status} == {"indexed", "unchanged"}
            assert first.stored.version_id == second.stored.version_id
            async with sessions() as session:
                assert await session.scalar(select(func.count()).select_from(DocumentVersion).where(
                    DocumentVersion.document_id == first.stored.document_id
                )) == 1
        finally:
            await engine.dispose()
    asyncio.run(scenario())


def test_invalid_input_fails_before_embedding(index_database_url):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            document = sample_document()
            with pytest.raises(ValueError, match="SHA-256"):
                await indexer.index(document, content_hash="invalid")
            document.chunks[0].metadata.document_id = "wrong identity"
            with pytest.raises(ValueError, match="identity"):
                await indexer.index(document, content_hash=digest("source"))
            assert provider.calls == 0
        finally:
            await engine.dispose()
    asyncio.run(scenario())


def test_chunk_write_failure_rolls_back_new_version(index_database_url):
    async def scenario():
        engine, sessions, provider, indexer = await setup(index_database_url)
        try:
            document = sample_document()
            first = await indexer.index(document, content_hash=digest("original"))

            def fail_chunk_write(connection, cursor, statement, parameters, context, executemany):
                if statement.startswith("INSERT INTO document_chunks"):
                    raise RuntimeError("Synthetic chunk write failure")

            event.listen(engine.sync_engine, "before_cursor_execute", fail_chunk_write)
            try:
                with pytest.raises(RuntimeError, match="chunk write"):
                    await indexer.index(document, content_hash=digest("changed"))
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", fail_chunk_write)
            async with sessions() as session:
                stored = await session.get(Document, first.stored.document_id)
                assert stored.active_version_id == first.stored.version_id
                assert await session.scalar(select(func.count()).select_from(DocumentVersion).where(
                    DocumentVersion.document_id == stored.id
                )) == 1
        finally:
            await engine.dispose()
    asyncio.run(scenario())
