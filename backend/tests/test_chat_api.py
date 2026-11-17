"""HTTP → real RAG pipeline/SDK mocks → isolated PostgreSQL; no paid calls."""

import asyncio
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
from openai import AsyncOpenAI
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1.chat import get_pipeline
from app.ingestion.embedding import OpenAIEmbeddingProvider
import app.main as application
from app.models import Document, DocumentChunk, DocumentVersion
from app.rag.generator import OpenAIChatProvider


@pytest.fixture(scope="module")
def corpus(index_database_url):
    async def seed():
        engine = create_async_engine(index_database_url)
        try:
            async with async_sessionmaker(engine)() as session, session.begin():
                document = Document(title="Backend guide", source_path=f"chat/{uuid4()}.md", source_type="markdown")
                session.add(document)
                await session.flush()
                version = DocumentVersion(document_id=document.id, content_hash="a" * 64, version_number=1,
                                          status="ready", embedding_model="text-embedding-3-small")
                session.add(version)
                await session.flush()
                document.active_version_id = version.id
                chunk = DocumentChunk(document_version_id=version.id, chunk_index=0,
                                      content="Synthetic searchable evidence",
                                      chunk_metadata={"heading_path": ["Example"], "source_line_ranges": [{"start": 8, "end": 9}],
                                                      "document_metadata": {"source_url": "https://example.org/guide"}},
                                      embedding=[1.0] + [0.0] * 1535)
                session.add(chunk)
                await session.flush()
                return str(chunk.id), str(version.id)
        finally:
            await engine.dispose()
    return asyncio.run(seed())


@pytest.fixture
def client(index_database_url, corpus, monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", index_database_url.database)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-chat-key")
    calls = {"embedding": [], "generation": [], "order": []}
    state = {"stage": "generation", "failure": None, "answer": "Supported answer [S1]"}

    def handler(request):
        stage = "embedding" if request.url.path.endswith("/embeddings") else "generation"
        payload = json.loads(request.content)
        calls[stage].append(payload)
        calls["order"].append(stage)
        failure = state["failure"] if state["stage"] == stage else None
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE_UPSTREAM_BODY", request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "PRIVATE_UPSTREAM_BODY", "type": "synthetic_error"}})
        if stage == "embedding":
            return httpx.Response(200, json={
                "object": "list", "model": payload["model"],
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0] + [0.0] * 1535}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5},
            })
        text = "PRIVATE_GENERATED_TEXT [S999]" if failure == "invalid_marker" else state["answer"]
        content = [{"type": "output_text", "text": "" if failure == "empty" else text, "annotations": []}]
        if failure == "refusal":
            content = [{"type": "refusal", "refusal": "Synthetic model refusal"}]
        return httpx.Response(200, json={
            "id": "resp_chat_synthetic", "object": "response", "created_at": 0,
            "status": "incomplete" if failure == "incomplete" else "completed",
            "model": payload["model"],
            "output": [{"id": "msg_synthetic", "type": "message", "role": "assistant", "status": "completed", "content": content}],
            "usage": {"input_tokens": 200, "output_tokens": 30, "total_tokens": 230,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        })

    def sdk():
        return AsyncOpenAI(api_key="synthetic-chat-key", max_retries=0, base_url="https://example.invalid/v1",
                           http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(application, "OpenAIEmbeddingProvider", lambda settings: OpenAIEmbeddingProvider(settings, client=sdk()))
    monkeypatch.setattr(application, "OpenAIChatProvider", lambda settings: OpenAIChatProvider(settings, client=sdk()))
    with TestClient(application.app) as test_client:
        yield test_client, calls, state


def test_real_pipeline_response_has_sources_usage_and_no_internal_prompt(client, corpus):
    test_client, calls, _ = client
    response = test_client.post("/api/v1/chat", json={"message": "  Synthetic question  ", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Synthetic question" and data["answer"] == "Supported answer [S1]"
    assert data["status"] == "generated" and data["citations"]["status"] == "valid"
    assert data["citations"]["grounding_checked"] is False
    assert data["sources"][0]["chunk_id"] == corpus[0] and data["sources"][0]["document_version_id"] == corpus[1]
    assert data["sources"][0]["title"] == "Backend guide"
    assert data["sources"][0]["source_line_ranges"] == [{"start": 8, "end": 9}]
    assert data["retrieval"]["chunks"][0]["included_in_context"] is True
    assert data["usage"]["embedding"]["total_tokens"] == 5
    assert data["usage"]["generation"]["total_tokens"] == 230
    assert data["chat_model"] == "gpt-6-luna" and calls["order"] == ["embedding", "generation"]
    assert "prompt" not in data and "generation" not in data and "messages" not in data
    assert calls["generation"][0]["store"] is False


@pytest.mark.parametrize("payload", [
    {}, {"message": " "}, {"message": "question", "top_k": 0},
    {"message": "question", "top_k": True}, {"message": "question", "top_k": "5"},
    {"message": "question", "system": "override rules"},
])
def test_invalid_requests_are_422_without_model_calls(client, payload):
    test_client, calls, _ = client
    assert test_client.post("/api/v1/chat", json=payload).status_code == 422
    assert calls["order"] == []


def test_question_limit_is_400_before_query_embedding(client):
    test_client, calls, _ = client
    response = test_client.post("/api/v1/chat", json={"message": "word " * 1500})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "input_limit_exceeded"
    assert calls["order"] == []


@pytest.mark.parametrize(("stage", "failure", "status", "code"), [
    ("embedding", 429, 503, "model_rate_limit"),
    ("generation", 429, 503, "model_rate_limit"),
    ("generation", 401, 502, "model_request_failed"),
    ("generation", "timeout", 504, "model_timeout"),
    ("generation", "incomplete", 502, "generation_failed"),
    ("generation", "empty", 502, "generation_failed"),
    ("generation", "invalid_marker", 502, "invalid_citations"),
])
def test_failed_generation_is_safe_and_not_a_partial_answer(client, stage, failure, status, code):
    test_client, _, state = client
    state.update(stage=stage, failure=failure)
    response = test_client.post("/api/v1/chat", json={"message": "question"})
    assert response.status_code == status and response.json()["detail"]["code"] == code
    assert "answer" not in response.json()
    assert "PRIVATE" not in response.text and "synthetic-chat-key" not in response.text
    assert "S999" not in response.text


def test_no_context_is_200_without_answer_model_usage(client):
    test_client, calls, _ = client
    application.app.state.embedding_provider.model = "absent-model"
    response = test_client.post("/api/v1/chat", json={"message": "question"})
    data = response.json()
    assert response.status_code == 200 and data["status"] == "insufficient_context"
    assert data["sources"] == [] and data["chat_model"] is None
    assert data["usage"]["generation"] is None and calls["generation"] == []
    assert data["citations"]["status"] == "not_applicable"


def test_missing_citations_remains_visible(client):
    test_client, _, state = client
    state["answer"] = "Sources do not contain enough information."
    data = test_client.post("/api/v1/chat", json={"message": "question"}).json()
    assert data["citations"]["status"] == "missing" and data["sources"] == []


def test_explicit_model_refusal_is_200_and_distinct(client):
    test_client, _, state = client
    state["failure"] = "refusal"
    response = test_client.post("/api/v1/chat", json={"message": "question"})
    assert response.status_code == 200 and response.json()["status"] == "model_refused"
    assert response.json()["citations"]["status"] == "not_applicable"


def test_missing_chat_provider_is_503_and_health_is_available(client):
    test_client, calls, _ = client
    application.app.state.chat_provider = None
    assert test_client.get("/health").status_code == 200
    assert test_client.post("/api/v1/chat", json={"message": "question"}).status_code == 503
    assert calls["order"] == []


def test_database_failure_is_503_without_private_error_details(client):
    class UnavailablePipeline:
        async def ask(self, *args, **kwargs):
            raise OperationalError("PRIVATE SQL", {"private": "PRIVATE DATA"}, Exception("synthetic"))

    application.app.dependency_overrides[get_pipeline] = lambda: UnavailablePipeline()
    try:
        response = client[0].post("/api/v1/chat", json={"message": "question"})
        assert response.status_code == 503 and "PRIVATE" not in response.text
    finally:
        application.app.dependency_overrides.clear()


def test_chat_is_in_openapi_without_internal_rag_result(client):
    specification = client[0].get("/openapi.json").json()
    assert "post" in specification["paths"]["/api/v1/chat"]
    assert "PreparedPrompt" not in specification["components"]["schemas"]
    assert "RAGResult" not in specification["components"]["schemas"]
