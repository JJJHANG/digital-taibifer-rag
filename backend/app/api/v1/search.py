"""Expose retrieval results with safe dependency error translation."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError
from sqlalchemy.exc import SQLAlchemyError

from app.ingestion.embedding import EmbeddingInputError
from app.rag.retriever import PgVectorRetriever
from app.rag.schemas import RetrievalRequest, RetrievalResult
from app.repositories.chunk_search import ChunkSearchRepository


router = APIRouter(prefix="/api/v1", tags=["search"])


def get_retriever(request: Request) -> PgVectorRetriever:
    provider = request.app.state.embedding_provider
    if provider is None:
        raise HTTPException(status_code=503, detail="Embedding service is not configured")
    return PgVectorRetriever(provider, ChunkSearchRepository(request.app.state.session_factory))


@router.post(
    "/search", response_model=RetrievalResult,
    summary="Retrieve knowledge chunks by semantic similarity",
    responses={
        400: {"description": "Query exceeds embedding input limits"},
        502: {"description": "Embedding request or retrieval response failed"},
        503: {"description": "Embedding or database service is unavailable"},
        504: {"description": "Embedding request timed out"},
    },
)
async def search(
    body: RetrievalRequest,
    retriever: Annotated[PgVectorRetriever, Depends(get_retriever)],
) -> RetrievalResult:
    """Return active, ready, model-compatible chunks; no answer generation.

    score is cosine similarity (1 - distance), not confidence or probability.
    An empty results list is a successful search with no eligible chunks.
    """
    try:
        return await retriever.retrieve(body.query, top_k=body.top_k)
    except EmbeddingInputError:
        raise HTTPException(status_code=400, detail="Query is empty or exceeds embedding token limits") from None
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Embedding request timed out") from None
    except RateLimitError:
        raise HTTPException(status_code=503, detail="Embedding service rate or quota limit reached") from None
    except APIConnectionError:
        raise HTTPException(status_code=503, detail="Embedding service is unavailable") from None
    except OpenAIError:
        raise HTTPException(status_code=502, detail="Embedding request failed") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Database service is unavailable") from None
    except ValueError:
        raise HTTPException(status_code=502, detail="Retrieval response failed validation") from None
