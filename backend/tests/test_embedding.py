"""Mock HTTP transport verifies real SDK serialization without external calls."""

import asyncio
import json

import httpx
from openai import AsyncOpenAI, RateLimitError
import pytest

from app.core.config import EmbeddingSettings
from app.ingestion.embedding import OpenAIEmbeddingProvider


def provider_for(handler, **settings):
    client = AsyncOpenAI(
        api_key="synthetic-test-key", base_url="https://example.invalid/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return OpenAIEmbeddingProvider(EmbeddingSettings(**settings), client=client)


def response_for(request, *, fault=None):
    payload = json.loads(request.content)
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i + 1)] + [0.0] * 1535}
        for i in range(len(payload["input"]))
    ]
    if fault == "count":
        data.pop()
    elif fault == "duplicate":
        data[1]["index"] = 0
    elif fault == "dimensions":
        data[0]["embedding"] = [1.0, 2.0]
    elif fault == "zero":
        data[0]["embedding"] = [0.0] * 1536
    return httpx.Response(200, json={
        "object": "list", "model": "wrong-model" if fault == "model" else payload["model"],
        "data": list(reversed(data)), "usage": {"prompt_tokens": 4, "total_tokens": 4},
    })


def test_batching_request_options_response_order_and_usage():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response_for(request)

    async def scenario():
        async with provider_for(handler, embedding_batch_size=2) as provider:
            result = await provider.embed_documents(["first", "second", "third"])
        assert [len(request["input"]) for request in requests] == [2, 1]
        assert all(request["dimensions"] == 1536 and request["encoding_format"] == "float" for request in requests)
        assert [vector[0] for vector in result.vectors] == [1, 2, 1]
        assert result.usage.prompt_tokens == result.usage.total_tokens == 8
        assert result.usage.requests == 2

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["count", "duplicate", "dimensions", "zero", "model"])
def test_invalid_provider_responses_are_rejected(fault):
    async def scenario():
        async with provider_for(lambda request: response_for(request, fault=fault)) as provider:
            with pytest.raises(ValueError):
                await provider.embed_documents(["first", "second"])
    asyncio.run(scenario())


def test_all_inputs_validated_before_any_network_call():
    requests = []

    async def scenario():
        async with provider_for(lambda request: requests.append(request), embedding_batch_size=1) as provider:
            with pytest.raises(ValueError, match="empty"):
                await provider.embed_documents(["valid", "   "])
            with pytest.raises(ValueError, match="token limit"):
                await provider.embed_documents(["valid", "a " * 9000])
            assert requests == []
            assert (await provider.embed_documents([])).vectors == []
    asyncio.run(scenario())


def test_api_errors_propagate_without_fake_vector_fallback():
    async def scenario():
        handler = lambda request: httpx.Response(429, json={"error": {"message": "Synthetic rate limit", "type": "rate_limit"}})
        async with provider_for(handler) as provider:
            with pytest.raises(RateLimitError):
                await provider.embed_documents(["sample"])
    asyncio.run(scenario())


def test_total_token_budget_can_split_before_item_count_limit():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response_for(request)

    async def scenario():
        async with provider_for(handler, embedding_batch_size=10) as provider:
            provider.MAX_BATCH_TOKENS = 2
            result = await provider.embed_documents(["first", "second", "third"])
            assert len(requests) == 2
            assert len(result.vectors) == 3
    asyncio.run(scenario())


def test_missing_credentials_fail_without_exposing_secrets():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(EmbeddingSettings(openai_api_key=""))


def test_dimensions_are_parsed_from_environment_and_schema_locked(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1536")
    assert EmbeddingSettings().embedding_dimensions == 1536
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "3072")
    with pytest.raises(ValueError):
        EmbeddingSettings()
