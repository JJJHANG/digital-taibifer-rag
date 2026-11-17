"""Stateless Responses API text generation with explicit input/output limits."""

from types import TracebackType
from typing import Protocol, Self

from openai import AsyncOpenAI
import tiktoken

from app.core.config import ChatSettings
from app.rag.schemas import ChatUsage, GenerationResult, PreparedPrompt


class GenerationError(RuntimeError):
    """Safe failure code; never includes provider body or source text."""

    def __init__(self, code: str, *, usage: ChatUsage | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.usage = usage


class ChatProvider(Protocol):
    model: str
    tokenizer: str
    max_input_tokens: int

    async def generate(self, prompt: PreparedPrompt) -> GenerationResult: ...


class OpenAIChatProvider:
    """Own the SDK client, including injected test clients, until context exit."""

    def __init__(self, settings: ChatSettings, *, client: AsyncOpenAI | None = None) -> None:
        self.model = settings.chat_model
        self.tokenizer = settings.chat_tokenizer
        self.max_input_tokens = settings.chat_max_input_tokens
        self._settings = settings
        if client is None:
            key = settings.openai_api_key
            if key is None or not key.get_secret_value().strip():
                raise ValueError("Set OPENAI_API_KEY in the ignored .env before using generation")
            client = AsyncOpenAI(
                api_key=key.get_secret_value(), base_url="https://api.openai.com/v1",
                timeout=settings.chat_timeout_seconds, max_retries=settings.chat_max_retries,
            )
        self._client = client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, traceback: TracebackType | None) -> None:
        await self._client.close()

    async def generate(self, prompt: PreparedPrompt) -> GenerationResult:
        if prompt.tokenizer != self.tokenizer:
            raise ValueError("Prompt tokenizer differs from the Chat Provider estimate")
        if [message.role for message in prompt.messages] != ["system", "user"]:
            raise ValueError("Expected one system message and one user JSON message")
        encoding = tiktoken.get_encoding(self.tokenizer)
        tokens = sum(len(encoding.encode(message.content, disallowed_special=())) for message in prompt.messages)
        if tokens > self.max_input_tokens:
            raise ValueError("Prompt exceeds the configured input token estimate")
        if tokens + self._settings.chat_framing_token_reserve + self._settings.chat_max_output_tokens > self._settings.chat_context_window:
            raise ValueError("Prompt and output allowance exceed the configured context window")
        response = await self._client.responses.create(
            model=self.model,
            input=[message.model_dump() for message in prompt.messages],
            max_output_tokens=self._settings.chat_max_output_tokens,
            reasoning={"effort": self._settings.chat_reasoning_effort},
            store=False, tools=[], tool_choice="none", truncation="disabled",
        )
        usage = None
        if response.usage is not None:
            usage = ChatUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
                cached_input_tokens=getattr(response.usage.input_tokens_details, "cached_tokens", 0) or 0,
                reasoning_tokens=getattr(response.usage.output_tokens_details, "reasoning_tokens", 0) or 0,
            )
        if response.status != "completed":
            raise GenerationError("response_not_completed", usage=usage)
        if usage is None:
            raise GenerationError("missing_usage")
        if not (response.model == self.model or response.model.startswith(self.model + "-")):
            raise GenerationError("model_mismatch", usage=usage)
        if any(item.type not in {"message", "reasoning"} for item in response.output):
            raise GenerationError("unexpected_output_type", usage=usage)
        refusals = [part.refusal for item in response.output if item.type == "message"
                    for part in item.content if part.type == "refusal"]
        text = response.output_text.strip()
        if refusals:
            if text:
                raise GenerationError("mixed_refusal_and_text", usage=usage)
            text = "\n".join(refusals).strip()
        if not text:
            raise GenerationError("empty_response", usage=usage)
        return GenerationResult(
            text=text, status="refused" if refusals else "completed", requested_model=self.model,
            model=response.model, response_id=response.id, usage=usage,
        )
