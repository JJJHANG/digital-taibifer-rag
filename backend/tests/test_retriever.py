"""Exact ranking, version isolation, and query validation with synthetic data."""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.ingestion.embedding import EmbeddingBatch, EmbeddingUsage
from app.models import Document, DocumentChunk, DocumentVersion
from app.rag.retriever import PgVectorRetriever
from app.rag.schemas import RetrievalResult
from app.repositories.chunk_search import ChunkSearchRepository


def vector(x, y=0):
    return [float(x), float(y)] + [0.0] * 1534


async def seed_version(session, document, number, *, model="text-embedding-3-small", status="ready", vectors=None):
    version = DocumentVersion(document_id=document.id, content_hash="a" * 64, version_number=number,
                              status=status, embedding_model=model)
    session.add(version)
    await session.flush()
    for index, embedding in enumerate(vectors or [vector(1)]):
        session.add(DocumentChunk(document_version_id=version.id, chunk_index=index,
                                  content=f"Synthetic version {number} chunk {index}",
                                  chunk_metadata={"heading_path": ["Example"], "page_number": 1}, embedding=embedding))
    await session.flush()
    return version


async def seed_document(session):
    document = Document(title="Synthetic guide", source_path=f"test/{uuid4()}.md", source_type="markdown")
    session.add(document)
    await session.flush()
    return document


class FakeQueryProvider:
    model = "text-embedding-3-small"
    dimensions = 1536

    def __init__(self):
        self.inputs = []
        self.result = EmbeddingBatch(model=self.model, dimensions=self.dimensions, vectors=[vector(1)],
                                     usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3, requests=1))

    async def embed_documents(self, texts):
        self.inputs.append(texts)
        return self.result


def test_exact_order_and_active_ready_model_filters(index_database_url):
    async def scenario():
        engine = create_async_engine(index_database_url)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with sessions() as session, session.begin():
                document = await seed_document(session)
                old = await seed_version(session, document, 1)
                active = await seed_version(session, document, 2, vectors=[vector(1), vector(1, 1), vector(0, 1)])
                document.active_version_id = active.id
                failed = await seed_document(session)
                failed.active_version_id = (await seed_version(session, failed, 1, status="failed")).id
                incompatible = await seed_document(session)
                incompatible.active_version_id = (await seed_version(session, incompatible, 1, model="other-model")).id
                unpublished = await seed_document(session)
                await seed_version(session, unpublished, 1)
            provider = FakeQueryProvider()
            result = await PgVectorRetriever(provider, ChunkSearchRepository(sessions)).retrieve("  example query  ", top_k=20)
            assert provider.inputs == [["example query"]]
            assert len(result.results) == 3
            assert [chunk.chunk_index for chunk in result.results] == [0, 1, 2]
            assert [chunk.score for chunk in result.results] == pytest.approx([1, 2**-0.5, 0], abs=1e-6)
            assert all(chunk.document_version_id == active.id and chunk.document_version_id != old.id for chunk in result.results)
            assert result.results[0].metadata["heading_path"] == ["Example"]
            assert result.usage.total_tokens == 3
            assert RetrievalResult.model_validate_json(result.model_dump_json()) == result
            limited = await PgVectorRetriever(provider, ChunkSearchRepository(sessions)).retrieve("query", top_k=1)
            assert len(limited.results) == 1 and limited.results[0].chunk_index == 0
        finally:
            await engine.dispose()
    asyncio.run(scenario())


def test_empty_results_and_query_does_not_write(index_database_url):
    async def scenario():
        engine = create_async_engine(index_database_url)
        sessions = async_sessionmaker(engine)
        try:
            # A different model sees no eligible vectors even when rows exist.
            provider = FakeQueryProvider()
            provider.model = provider.result.model = "absent-model"
            result = await PgVectorRetriever(provider, ChunkSearchRepository(sessions)).retrieve("question")
            assert result.results == [] and result.usage.requests == 1
        finally:
            await engine.dispose()
    asyncio.run(scenario())


@pytest.mark.parametrize(("query", "top_k"), [("", 5), ("   ", 5), ("query", 0), ("query", 101), ("query", True), ("query", 2.5)])
def test_bad_request_fails_before_api(query, top_k):
    async def scenario():
        provider = FakeQueryProvider()
        with pytest.raises(ValueError):
            await PgVectorRetriever(provider, None).retrieve(query, top_k=top_k)
        assert provider.inputs == []
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["dimensions", "response_model", "missing", "zero", "nan"])
def test_bad_query_vectors_never_reach_database(fault):
    async def scenario():
        provider = FakeQueryProvider()
        if fault == "dimensions":
            provider.dimensions = 3
        elif fault == "response_model":
            provider.result.model = "wrong-model"
        elif fault == "missing":
            provider.result.vectors = []
        elif fault == "zero":
            provider.result.vectors = [[0.0] * 1536]
        else:
            provider.result.vectors[0][0] = float("nan")
        with pytest.raises(ValueError):
            await PgVectorRetriever(provider, None).retrieve("question")
    asyncio.run(scenario())
