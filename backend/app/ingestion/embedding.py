"""OpenAI embeddings with bounded batches, usage, and vector integrity checks."""

import math
from types import TracebackType
from typing import Protocol, Self

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import tiktoken

from app.core.config import EmbeddingSettings


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)


class EmbeddingBatch(BaseModel):
    model: str
    dimensions: int = Field(gt=0)
    vectors: list[list[float]]
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


class EmbeddingPlan(BaseModel):
    token_counts: list[int]
    # Zero-based, end-exclusive input indices for each request.
    batches: list[tuple[int, int]]


class EmbeddingInputError(ValueError):
    """Text cannot be submitted to the embedding model without changing it."""


def plan_embedding_batches(
    texts: list[str], *, batch_size: int,
    max_input_tokens: int = 8191, max_batch_tokens: int = 300_000,
) -> EmbeddingPlan:
    """Shared by dry-run and the real provider; makes no API or DB calls."""
    if not 1 <= batch_size <= 2048 or min(max_input_tokens, max_batch_tokens) <= 0:
        raise ValueError("Invalid embedding batch limits")
    encoding = tiktoken.get_encoding("cl100k_base")
    counts: list[int] = []
    for position, text in enumerate(texts):
        if not text.strip():
            raise EmbeddingInputError(f"Embedding input {position} is empty or whitespace-only")
        count = len(encoding.encode(text, disallowed_special=()))
        if count > min(max_input_tokens, max_batch_tokens):
            raise EmbeddingInputError(f"Embedding input {position} exceeds the token limit; rechunk it")
        counts.append(count)
    batches: list[tuple[int, int]] = []
    start = 0
    while start < len(texts):
        end, tokens = start, 0
        while end < len(texts) and end - start < batch_size:
            if tokens + counts[end] > max_batch_tokens:
                break
            tokens += counts[end]
            end += 1
        batches.append((start, end))
        start = end
    return EmbeddingPlan(token_counts=counts, batches=batches)


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed_documents(self, texts: list[str]) -> EmbeddingBatch: ...


def validate_vectors(vectors: list[list[float]], *, count: int, dimensions: int) -> None:
    """Reject partial, nonfinite, and zero-norm results before DB writes."""
    if len(vectors) != count:
        raise ValueError("Embedding count does not match input count")
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("Embedding dimensions do not match the database schema")
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise ValueError("Embedding must be finite and have a nonzero norm")


class OpenAIEmbeddingProvider:
    """Own an async SDK client; injected clients are closed on context exit too."""

    MAX_INPUT_TOKENS = 8191  # Conservative bound below the API's 8192 limit.
    MAX_BATCH_TOKENS = 300_000

    def __init__(self, settings: EmbeddingSettings, *, client: AsyncOpenAI | None = None) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._batch_size = settings.embedding_batch_size
        if client is None:
            key = settings.openai_api_key
            if key is None or not key.get_secret_value().strip():
                raise ValueError("Set OPENAI_API_KEY in the ignored .env before using embeddings")
            client = AsyncOpenAI(
                api_key=key.get_secret_value(),
                base_url="https://api.openai.com/v1",
                timeout=settings.embedding_timeout_seconds,
                max_retries=settings.embedding_max_retries,
            )
        self._client = client

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None,
        exc: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        await self._client.close()

    async def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        plan = plan_embedding_batches(
            texts, batch_size=self._batch_size,
            max_input_tokens=self.MAX_INPUT_TOKENS, max_batch_tokens=self.MAX_BATCH_TOKENS,
        )
        vectors: list[list[float]] = []
        usage = EmbeddingUsage()
        for start, end in plan.batches:
            response = await self._client.embeddings.create(
                model=self.model, dimensions=self.dimensions,
                input=texts[start:end], encoding_format="float",
            )
            if response.model != self.model:
                raise ValueError("Embedding response model differs from the requested model")
            if sorted(item.index for item in response.data) != list(range(end - start)):
                raise ValueError("Embedding response has missing or duplicate input indices")
            ordered = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            validate_vectors(ordered, count=end - start, dimensions=self.dimensions)
            vectors.extend(ordered)
            usage.prompt_tokens += response.usage.prompt_tokens
            usage.total_tokens += response.usage.total_tokens
            usage.requests += 1
        return EmbeddingBatch(model=self.model, dimensions=self.dimensions, vectors=vectors, usage=usage)
