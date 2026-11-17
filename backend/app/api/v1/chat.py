"""HTTP chat boundary for the existing retrieval/generation/citation pipeline."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.chat_schemas import (
    ChatCitationStatus, ChatRequest, ChatResponse, ChatRetrievalHit, ChatRetrievalSummary, ChatUsageSummary,
)
from app.ingestion.embedding import EmbeddingInputError
from app.rag.citations import CitationContextError, CitationValidationError
from app.rag.generator import GenerationError
from app.rag.pipeline import RAGPipeline
from app.rag.prompt import PromptInputError
from app.rag.retriever import PgVectorRetriever
from app.repositories.chunk_search import ChunkSearchRepository


router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_pipeline(request: Request) -> RAGPipeline:
    embedding = request.app.state.embedding_provider
    generator = request.app.state.chat_provider
    if embedding is None or generator is None:
        raise HTTPException(status_code=503, detail={
            "code": "model_service_not_configured", "message": "Embedding or chat service is not configured",
        })
    retriever = PgVectorRetriever(embedding, ChunkSearchRepository(request.app.state.session_factory))
    return RAGPipeline(retriever, generator)


@router.post(
    "/chat", response_model=ChatResponse,
    summary="Answer from indexed knowledge with source identity checks",
    responses={
        400: {"description": "Message exceeds RAG input limits"},
        502: {"description": "Generation, context or citation validation failed"},
        503: {"description": "Model or database service is unavailable"},
        504: {"description": "Model request timed out"},
    },
)
async def chat(
    body: ChatRequest, pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> ChatResponse:
    """Single-turn JSON response; no conversation storage or streaming.

    sources are backend mappings for cited evidence, not model-provided URLs.
    citations.status=valid checks identity only; grounding_checked stays false.
    Missing markers are explicit and return no authoritative sources.
    """
    try:
        result = await pipeline.ask(body.message, top_k=body.top_k)
    except (PromptInputError, EmbeddingInputError):
        raise HTTPException(status_code=400, detail={
            "code": "input_limit_exceeded", "message": "Message exceeds supported RAG input limits",
        }) from None
    except CitationValidationError:
        raise HTTPException(status_code=502, detail={
            "code": "invalid_citations", "message": "Generated answer contains invalid source markers",
        }) from None
    except CitationContextError:
        raise HTTPException(status_code=502, detail={
            "code": "invalid_source_context", "message": "Retrieved sources failed context validation",
        }) from None
    except GenerationError:
        raise HTTPException(status_code=502, detail={
            "code": "generation_failed", "message": "Model response was incomplete or invalid",
        }) from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "model_timeout", "message": "Model request timed out",
        }) from None
    except RateLimitError:
        raise HTTPException(status_code=503, detail={
            "code": "model_rate_limit", "message": "Model rate or quota limit reached",
        }) from None
    except APIConnectionError:
        raise HTTPException(status_code=503, detail={
            "code": "model_unavailable", "message": "Model service is unavailable",
        }) from None
    except OpenAIError:
        raise HTTPException(status_code=502, detail={
            "code": "model_request_failed", "message": "Model request failed",
        }) from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail={
            "code": "database_unavailable", "message": "Database service is unavailable",
        }) from None
    except ValueError:
        raise HTTPException(status_code=502, detail={
            "code": "invalid_pipeline_result", "message": "RAG response failed validation",
        }) from None

    selected = set(result.prompt.selected_chunk_ids)
    return ChatResponse(
        message=result.query, answer=result.answer, status=result.status,
        sources=result.citations.sources,
        citations=ChatCitationStatus(
            status=result.citations.status, cited_ids=result.citations.cited_ids,
            grounding_checked=result.citations.grounding_checked,
        ),
        retrieval=ChatRetrievalSummary(
            top_k=result.retrieval.top_k, retrieved_count=len(result.retrieval.results),
            context_count=len(selected), skipped_context_count=len(result.prompt.skipped_context),
            embedding_model=result.retrieval.embedding_model,
            chunks=[ChatRetrievalHit(
                chunk_id=chunk.chunk_id, document_id=chunk.document_id,
                document_version_id=chunk.document_version_id, score=chunk.score,
                included_in_context=chunk.chunk_id in selected,
            ) for chunk in result.retrieval.results],
        ),
        chat_model=result.generation.model if result.generation else None,
        prompt_version=result.prompt.prompt_version,
        usage=ChatUsageSummary(
            embedding=result.retrieval.usage,
            generation=result.generation.usage if result.generation else None,
        ),
    )
