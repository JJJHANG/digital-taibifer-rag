"""Pipeline flow and short circuits with independent synthetic components."""

import asyncio
from uuid import uuid4

import pytest

from app.ingestion.embedding import EmbeddingUsage
from app.rag.generator import GenerationError
from app.rag.citations import CitationValidationError
from app.rag.pipeline import INSUFFICIENT_CONTEXT_ANSWER, RAGPipeline
from app.rag.schemas import (
    ChatUsage, GenerationResult, PromptConfig, RAGResult, RetrievalResult, RetrievedChunk,
)


def chunk(content="Synthetic evidence"):
    return RetrievedChunk(chunk_id=uuid4(), document_id=uuid4(), document_version_id=uuid4(), version_number=1,
                          chunk_index=0, title="Synthetic guide", source_path="synthetic.md", source_type="markdown",
                          content=content, distance=0.2, score=0.8, metadata={"heading_path": ["Example"]})


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    async def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return RetrievalResult(query=query, top_k=top_k, embedding_model="text-embedding-3-small", results=self.chunks,
                               usage=EmbeddingUsage(prompt_tokens=3, total_tokens=3, requests=1))


class FakeGenerator:
    model, tokenizer, max_input_tokens = "gpt-6-luna", "o200k_base", 8000

    def __init__(self):
        self.calls = []
        self.failure = None
        self.status = "completed"
        self.text = "Synthetic answer [S1]"

    async def generate(self, prompt):
        self.calls.append(prompt)
        if self.failure:
            raise self.failure
        return GenerationResult(text=self.text, status=self.status, requested_model=self.model,
                                model=self.model, response_id="resp_synthetic",
                                usage=ChatUsage(input_tokens=100, output_tokens=20, total_tokens=120))


def test_retrieval_prompt_generation_and_separate_usage():
    async def scenario():
        source = chunk()
        retriever, generator = FakeRetriever([source]), FakeGenerator()
        result = await RAGPipeline(retriever, generator).ask("  question  ", top_k=3)
        assert result.status == "generated" and result.answer == "Synthetic answer [S1]"
        assert retriever.calls == [("question", 3)]
        assert result.prompt.selected_chunk_ids == [source.chunk_id]
        assert result.retrieval.usage.total_tokens == 3 and result.generation.usage.total_tokens == 120
        assert result.prompt.tokenizer == "o200k_base"
        assert result.citations.status == "valid" and result.citations.sources[0].chunk_id == source.chunk_id
        assert RAGResult.model_validate_json(result.model_dump_json()) == result
    asyncio.run(scenario())


@pytest.mark.parametrize("chunks", [[], [" "], ["evidence " * 1000]])
def test_no_usable_context_skips_answer_model(chunks):
    async def scenario():
        retriever, generator = FakeRetriever([chunk(text) for text in chunks]), FakeGenerator()
        pipeline = RAGPipeline(retriever, generator, config=PromptConfig(tokenizer="o200k_base", max_context_tokens=1))
        result = await pipeline.ask("question")
        assert result.status == "insufficient_context" and result.answer == INSUFFICIENT_CONTEXT_ANSWER
        assert result.generation is None and generator.calls == []
        assert result.citations.status == "not_applicable"
    asyncio.run(scenario())


def test_invalid_question_fails_before_retrieval():
    async def scenario():
        retriever, generator = FakeRetriever([]), FakeGenerator()
        pipeline = RAGPipeline(retriever, generator, config=PromptConfig(tokenizer="o200k_base", max_question_tokens=2))
        with pytest.raises(ValueError):
            await pipeline.ask("a " * 50)
        with pytest.raises(ValueError):
            await pipeline.ask("question", top_k=0)
        assert retriever.calls == [] and generator.calls == []
    asyncio.run(scenario())


def test_generation_failure_propagates_and_refusal_remains_distinct():
    async def scenario():
        retriever, generator = FakeRetriever([chunk()]), FakeGenerator()
        pipeline = RAGPipeline(retriever, generator)
        generator.failure = GenerationError("response_not_completed")
        with pytest.raises(GenerationError):
            await pipeline.ask("question")
        generator.failure, generator.status = None, "refused"
        assert (await pipeline.ask("question")).status == "model_refused"
    asyncio.run(scenario())


def test_incompatible_config_fails_before_any_work():
    with pytest.raises(ValueError):
        RAGPipeline(FakeRetriever([]), FakeGenerator(), config=PromptConfig())


def test_unknown_citation_blocks_answer_and_preserves_generation_usage():
    async def scenario():
        generator = FakeGenerator()
        generator.text = "Invented source [S999]"
        with pytest.raises(CitationValidationError) as caught:
            await RAGPipeline(FakeRetriever([chunk()]), generator).ask("question")
        assert caught.value.report.invalid_markers == ["[S999]"]
        assert caught.value.usage.total_tokens == 120
    asyncio.run(scenario())


def test_missing_markers_are_reported_without_guessing_answer_intent():
    async def scenario():
        generator = FakeGenerator()
        generator.text = "目前來源不足以完整回答。"
        result = await RAGPipeline(FakeRetriever([chunk()]), generator).ask("question")
        assert result.citations.status == "missing" and result.citations.sources == []
        assert result.answer == generator.text
    asyncio.run(scenario())
