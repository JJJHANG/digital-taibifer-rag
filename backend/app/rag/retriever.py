"""Embed a question, validate its vector, and retrieve source-bearing chunks."""

from app.ingestion.embedding import EmbeddingProvider, validate_vectors
from app.models.documents import EMBEDDING_DIMENSIONS
from app.rag.schemas import RetrievalRequest, RetrievalResult
from app.repositories.chunk_search import ChunkSearchRepository


class PgVectorRetriever:
    def __init__(self, provider: EmbeddingProvider, repository: ChunkSearchRepository) -> None:
        self._provider = provider
        self._repository = repository

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        request = RetrievalRequest(query=query, top_k=top_k)
        if self._provider.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError("Query provider dimensions differ from the database schema")
        # A query is one input to the same embedding model used during indexing.
        batch = await self._provider.embed_documents([request.query])
        if batch.model != self._provider.model or batch.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError("Query embedding model or dimensions do not match the provider")
        validate_vectors(batch.vectors, count=1, dimensions=EMBEDDING_DIMENSIONS)
        chunks = await self._repository.search(
            batch.vectors[0], model=batch.model, top_k=request.top_k,
        )
        return RetrievalResult(
            query=request.query, top_k=request.top_k, embedding_model=batch.model,
            results=chunks, usage=batch.usage,
        )
