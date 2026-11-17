"""Explicitly opt-in, billable API check using synthetic text and a scratch DB."""

import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path

from openai import OpenAIError
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import EmbeddingSettings
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.embedding import OpenAIEmbeddingProvider
from app.ingestion.indexer import EmbeddingIndexer
from app.ingestion.parsers import parse_document
from app.models import Document, DocumentChunk, DocumentVersion
from app.repositories.document_index import DocumentIndexRepository


@pytest.mark.skipif(os.getenv("RUN_LIVE_EMBEDDING") != "1", reason="Paid API check is opt-in")
def test_real_embeddings_and_atomic_publication(index_database_url, tmp_path):
    source = tmp_path / "synthetic.md"
    source.write_text("# Synthetic guide\nThis is a synthetic example for embedding integration testing.\n")
    parsed = parse_document(source, document_id="synthetic-live", source_path="synthetic.md")
    chunked = RecursiveChunker().split_document(parsed)
    content_hash = sha256(source.read_bytes()).hexdigest()

    async def scenario():
        engine = create_async_engine(index_database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with OpenAIEmbeddingProvider(EmbeddingSettings()) as provider:
                indexer = EmbeddingIndexer(provider, DocumentIndexRepository(sessions))
                first = await indexer.index(chunked, content_hash=content_hash)
                second = await indexer.index(chunked, content_hash=content_hash)
            assert first.stored.status == "indexed" and second.stored.status == "unchanged"
            assert first.stored.version_id == second.stored.version_id
            assert first.usage.requests == 1 and second.usage.requests == 0
            async with sessions() as session:
                document = await session.get(Document, first.stored.document_id)
                version = await session.get(DocumentVersion, first.stored.version_id)
                chunk = (await session.scalars(select(DocumentChunk))).one()
                assert document.active_version_id == version.id and version.status == "ready"
                assert chunk.content == chunked.chunks[0].content
                assert await session.scalar(select(func.vector_dims(DocumentChunk.embedding))) == 1536
                distance = await session.scalar(select(DocumentChunk.embedding.cosine_distance(chunk.embedding)))
                assert distance == pytest.approx(0, abs=1e-6)
                counts = {
                    "documents": await session.scalar(select(func.count()).select_from(Document)),
                    "versions": await session.scalar(select(func.count()).select_from(DocumentVersion)),
                    "chunks": await session.scalar(select(func.count()).select_from(DocumentChunk)),
                }
            assert counts == {"documents": 1, "versions": 1, "chunks": 1}
            return {
                "status": "passed", "model": provider.model, "dimensions": 1536,
                "usage": first.usage.model_dump(), "repeat_usage": second.usage.model_dump(),
                "database_counts": counts, "self_cosine_distance": float(distance),
                "private_documents_sent": 0, "database": "disposable; removed after test",
            }
        finally:
            await engine.dispose()

    try:
        report = asyncio.run(scenario())
    except OpenAIError as error:
        # Do not print API bodies, authorization headers or credential fragments.
        pytest.fail(f"Live API check failed: {type(error).__name__}; status={getattr(error, 'status_code', None)}", pytrace=False)
    if destination := os.getenv("LIVE_EMBEDDING_REPORT"):
        Path(destination).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
