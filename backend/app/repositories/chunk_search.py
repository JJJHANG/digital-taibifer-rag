"""Exact pgvector search restricted to active, ready, compatible versions."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.ingestion.embedding import validate_vectors
from app.models import Document, DocumentChunk, DocumentVersion
from app.models.documents import EMBEDDING_DIMENSIONS
from app.rag.schemas import RetrievedChunk


class ChunkSearchRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(
        self, vector: list[float], *, model: str, top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        validate_vectors([vector], count=1, dimensions=EMBEDDING_DIMENSIONS)
        distance = DocumentChunk.embedding.cosine_distance(vector).label("distance")
        statement = (
            select(
                DocumentChunk.id.label("chunk_id"), Document.id.label("document_id"),
                DocumentVersion.id.label("document_version_id"), DocumentVersion.version_number,
                DocumentChunk.chunk_index, Document.title, Document.source_path, Document.source_type,
                DocumentChunk.content, DocumentChunk.chunk_metadata.label("metadata"), distance,
            )
            .select_from(DocumentChunk)
            .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
            .join(Document, DocumentVersion.document_id == Document.id)
            .where(
                Document.active_version_id == DocumentVersion.id,
                DocumentVersion.status == "ready",
                DocumentVersion.embedding_model == model,
            )
            .order_by(distance, DocumentChunk.id)
            .limit(top_k)
        )
        # Open the DB session only after the query embedding is available.
        async with self._sessions() as session:
            rows = (await session.execute(statement)).mappings().all()
        return [RetrievedChunk(**row, score=1.0 - row["distance"]) for row in rows]
