"""API contracts with real retrieval/PostgreSQL and mock embedding HTTP."""

import asyncio
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
from openai import AsyncOpenAI
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.exc import OperationalError

from app.api.v1.search import get_retriever
from app.ingestion.embedding import OpenAIEmbeddingProvider
import app.main as application
from app.models import Document, DocumentChunk, DocumentVersion


@pytest.fixture
def client(index_database_url, monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", index_database_url.database)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    calls = []
    state = {"failure": None}

    def handler(request):
        calls.append(json.loads(request.content))
        if state["failure"] == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if state["failure"] == "connection":
            raise httpx.ConnectError("synthetic connection", request=request)
        if state["failure"] == "dimensions":
            embedding = [1.0]
        else:
            embedding = [1.0] + [0.0] * 1535
        if isinstance(state["failure"], int):
            return httpx.Response(state["failure"], json={"error": {"message": "PRIVATE_API_BODY", "type": "synthetic_error"}})
        return httpx.Response(200, json={
            "object": "list", "model": json.loads(request.content)["model"],
            "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        })

    def build(settings):
        sdk = AsyncOpenAI(api_key="synthetic-test-key", max_retries=0, base_url="https://example.invalid/v1",
                          http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        return OpenAIEmbeddingProvider(settings, client=sdk)

    monkeypatch.setattr(application, "OpenAIEmbeddingProvider", build)
    with TestClient(application.app) as test_client:
        yield test_client, calls, state


def test_search_http_to_embedding_to_real_postgres(client, index_database_url):
    async def seed():
        engine = create_async_engine(index_database_url)
        try:
            async with async_sessionmaker(engine)() as session, session.begin():
                document = Document(title="Synthetic API guide", source_path=f"api/{uuid4()}.md", source_type="markdown")
                session.add(document)
                await session.flush()
                version = DocumentVersion(document_id=document.id, content_hash="a" * 64, version_number=1,
                                          status="ready", embedding_model="text-embedding-3-small")
                session.add(version)
                await session.flush()
                document.active_version_id = version.id
                chunk = DocumentChunk(document_version_id=version.id, chunk_index=0, content="Synthetic searchable knowledge",
                                      chunk_metadata={"heading_path": ["Example"]}, embedding=[1.0] + [0.0] * 1535)
                session.add(chunk)
                await session.flush()
                return str(chunk.id), str(version.id)
        finally:
            await engine.dispose()

    chunk_id, version_id = asyncio.run(seed())
    test_client, calls, _ = client
    response = test_client.post("/api/v1/search", json={"query": "  sample question  ", "top_k": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "sample question" and result["top_k"] == 1
    assert result["results"][0]["chunk_id"] == chunk_id
    assert result["results"][0]["document_version_id"] == version_id
    assert result["results"][0]["score"] == pytest.approx(1)
    assert result["results"][0]["metadata"]["heading_path"] == ["Example"]
    assert result["usage"]["total_tokens"] == 4
    assert calls[0]["input"] == ["sample question"]


@pytest.mark.parametrize("payload", [
    {}, {"query": ""}, {"query": "  "}, {"query": "question", "top_k": 0},
    {"query": "question", "top_k": 101}, {"query": "question", "top_k": True},
    {"query": "question", "top_k": "5"},
])
def test_invalid_requests_are_422_without_api_calls(client, payload):
    test_client, calls, _ = client
    assert test_client.post("/api/v1/search", json=payload).status_code == 422
    assert calls == []


def test_token_limit_is_400_without_api_calls(client):
    test_client, calls, _ = client
    assert test_client.post("/api/v1/search", json={"query": "a " * 9000}).status_code == 400
    assert calls == []


@pytest.mark.parametrize(("failure", "status"), [(429, 503), (401, 502), (500, 502),
                                                   ("timeout", 504), ("connection", 503), ("dimensions", 502)])
def test_dependency_errors_are_safe(client, failure, status):
    test_client, _, state = client
    state["failure"] = failure
    response = test_client.post("/api/v1/search", json={"query": "question"})
    assert response.status_code == status
    assert "PRIVATE_API_BODY" not in response.text
    assert "synthetic-test-key" not in response.text


def test_database_failure_is_503(client):
    class UnavailableRetriever:
        async def retrieve(self, *args, **kwargs):
            raise OperationalError("PRIVATE SQL", {"private": "PRIVATE CONTENT"}, Exception("synthetic"))

    application.app.dependency_overrides[get_retriever] = lambda: UnavailableRetriever()
    try:
        response = client[0].post("/api/v1/search", json={"query": "question"})
        assert response.status_code == 503 and "PRIVATE" not in response.text
    finally:
        application.app.dependency_overrides.clear()


def test_missing_key_keeps_health_available(client):
    test_client, calls, _ = client
    application.app.state.embedding_provider = None
    assert test_client.get("/health").status_code == 200
    assert test_client.post("/api/v1/search", json={"query": "question"}).status_code == 503
    assert calls == []


def test_search_is_documented_in_openapi(client):
    specification = client[0].get("/openapi.json").json()
    operation = specification["paths"]["/api/v1/search"]["post"]
    assert operation["requestBody"]["required"] is True
    assert "200" in operation["responses"] and "422" in operation["responses"]
    assert specification["components"]["schemas"]["RetrievedChunk"]["properties"]["score"]["type"] == "number"


def test_no_eligible_chunks_is_200_with_empty_results(client):
    test_client, calls, _ = client
    application.app.state.embedding_provider.model = "absent-model"
    response = test_client.post("/api/v1/search", json={"query": "question"})
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["usage"]["requests"] == 1 and len(calls) == 1
