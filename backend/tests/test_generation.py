"""Real Responses SDK with mock HTTP; no paid API calls."""

import asyncio
import json

import httpx
from openai import AsyncOpenAI, RateLimitError
import pytest

from app.core.config import ChatSettings
from app.rag.generator import GenerationError, OpenAIChatProvider
from app.rag.prompt import build_rag_prompt
from app.rag.schemas import PromptConfig


def response_body(payload, *, fault=None):
    content = [{"type": "output_text", "text": "Synthetic answer [S1]", "annotations": []}]
    if fault == "refusal":
        content = [{"type": "refusal", "refusal": "Synthetic model refusal"}]
    elif fault == "empty":
        content[0]["text"] = "   "
    result = {
        "id": "resp_synthetic", "object": "response", "created_at": 0,
        "status": "incomplete" if fault == "incomplete" else "completed",
        "model": "wrong-model" if fault == "model" else payload["model"],
        "output": [{"id": "msg_synthetic", "type": "message", "role": "assistant", "status": "completed", "content": content}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                  "input_tokens_details": {"cached_tokens": 10}, "output_tokens_details": {"reasoning_tokens": 5}},
    }
    if fault == "usage":
        result["usage"] = None
    return result


def provider_for(handler, **settings):
    client = AsyncOpenAI(api_key="synthetic-key", max_retries=0, base_url="https://example.invalid/v1",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return OpenAIChatProvider(ChatSettings(**settings), client=client)


def prompt():
    return build_rag_prompt("Synthetic question", [], config=PromptConfig(tokenizer="o200k_base"))


def test_response_options_text_and_usage():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(200, json=response_body(payload))

    async def scenario():
        async with provider_for(handler) as provider:
            result = await provider.generate(prompt())
        payload = requests[0]
        assert payload["model"] == "gpt-6-luna"
        assert payload["store"] is False and payload["tools"] == [] and payload["tool_choice"] == "none"
        assert payload["truncation"] == "disabled" and payload["reasoning"]["effort"] == "none"
        assert payload["max_output_tokens"] == 1500
        assert result.text == "Synthetic answer [S1]" and result.status == "completed"
        assert result.usage.input_tokens == 100 and result.usage.total_tokens == 120
        assert result.usage.cached_input_tokens == 10 and result.usage.reasoning_tokens == 5
    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["incomplete", "empty", "model", "usage"])
def test_invalid_responses_never_become_successful_answers(fault):
    async def scenario():
        async with provider_for(lambda request: httpx.Response(200, json=response_body(json.loads(request.content), fault=fault))) as provider:
            with pytest.raises(GenerationError) as caught:
                await provider.generate(prompt())
            assert "Synthetic answer" not in str(caught.value)
            if fault == "incomplete":
                assert caught.value.usage.total_tokens == 120
    asyncio.run(scenario())


def test_explicit_refusal_is_not_mistaken_for_empty_output():
    async def scenario():
        async with provider_for(lambda request: httpx.Response(200, json=response_body(json.loads(request.content), fault="refusal"))) as provider:
            result = await provider.generate(prompt())
            assert result.status == "refused" and result.text == "Synthetic model refusal"
    asyncio.run(scenario())


def test_sdk_error_propagates_without_fabricated_answer():
    async def scenario():
        async with provider_for(lambda request: httpx.Response(429, json={"error": {"message": "synthetic rate limit", "type": "rate_limit"}})) as provider:
            with pytest.raises(RateLimitError):
                await provider.generate(prompt())
    asyncio.run(scenario())


def test_input_checks_precede_network_calls():
    calls = []

    async def scenario():
        async with provider_for(lambda request: calls.append(request), chat_max_input_tokens=1) as provider:
            with pytest.raises(ValueError, match="exceeds"):
                await provider.generate(prompt())
        async with provider_for(lambda request: calls.append(request)) as provider:
            other_tokenizer = build_rag_prompt("question", [])
            with pytest.raises(ValueError, match="tokenizer"):
                await provider.generate(other_tokenizer)
        assert calls == []
    asyncio.run(scenario())


def test_missing_key_and_invalid_context_allowance():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChatProvider(ChatSettings(openai_api_key=""))
    with pytest.raises(ValueError):
        ChatSettings(chat_context_window=100)
