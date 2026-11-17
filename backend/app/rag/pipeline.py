"""Compose retrieval, context selection, and generation without persistence."""

from typing import Protocol

from app.rag.citations import build_citation_map, CitationValidationError, validate_citations
from app.rag.generator import ChatProvider
from app.rag.prompt import build_rag_prompt
from app.rag.schemas import PromptConfig, RAGResult, RetrievalRequest, RetrievalResult


INSUFFICIENT_CONTEXT_ANSWER = "目前知識庫沒有足夠資訊回答這個問題。"


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult: ...


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: ChatProvider, *, config: PromptConfig | None = None) -> None:
        self._retriever = retriever
        self._generator = generator
        self._config = config or PromptConfig(tokenizer=generator.tokenizer, max_input_tokens=generator.max_input_tokens)
        if self._config.tokenizer != generator.tokenizer or self._config.max_input_tokens > generator.max_input_tokens:
            raise ValueError("Prompt configuration is incompatible with the Chat Provider")

    async def ask(self, question: str, top_k: int = 5) -> RAGResult:
        request = RetrievalRequest(query=question, top_k=top_k)
        # Validate question and fixed rules before spending Query Embedding tokens.
        build_rag_prompt(request.query, [], config=self._config)
        retrieval = await self._retriever.retrieve(request.query, top_k=request.top_k)
        prompt = build_rag_prompt(request.query, retrieval.results, config=self._config)
        source_map = build_citation_map(prompt, retrieval.results)
        if not prompt.selected_chunk_ids:
            return RAGResult(
                query=request.query, answer=INSUFFICIENT_CONTEXT_ANSWER, status="insufficient_context",
                retrieval=retrieval, prompt=prompt, generation=None,
            )
        generation = await self._generator.generate(prompt)
        citations = validate_citations(generation.text, source_map, expected=generation.status == "completed")
        if citations.status == "invalid":
            raise CitationValidationError(citations, usage=generation.usage)
        return RAGResult(
            query=request.query, answer=generation.text,
            status="model_refused" if generation.status == "refused" else "generated",
            retrieval=retrieval, prompt=prompt, generation=generation, citations=citations,
        )
