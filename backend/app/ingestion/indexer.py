"""Embed outside DB transactions, then atomically publish a complete index."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from app.ingestion.embedding import EmbeddingProvider, EmbeddingUsage
from app.ingestion.schemas import ChunkedDocument
from app.models.documents import EMBEDDING_DIMENSIONS
from app.repositories.document_index import DocumentIndexRepository, StoredIndex


@dataclass(frozen=True)
class IndexResult:
    stored: StoredIndex
    usage: EmbeddingUsage


class EmbeddingIndexer:
    def __init__(self, provider: EmbeddingProvider, repository: DocumentIndexRepository) -> None:
        self._provider = provider
        self._repository = repository

    async def index(self, document: ChunkedDocument, *, content_hash: str) -> IndexResult:
        """content_hash is the caller's SHA-256 of the prepared input bytes.

        The extra fingerprint covers metadata, chunking, model and dimensions;
        matching raw bytes alone must not silently reuse an incompatible index.
        """
        # Keep the fingerprint, API inputs and DB snapshot consistent across awaits.
        document = document.model_copy(deep=True)
        if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        if self._provider.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError("Provider dimensions differ from the database schema")
        if not document.chunks or [chunk.chunk_index for chunk in document.chunks] != list(range(len(document.chunks))):
            raise ValueError("Chunks must be nonempty and consecutively indexed")
        first = document.chunks[0].metadata
        for chunk in document.chunks:
            if (
                not chunk.content.strip() or chunk.metadata.document_id != document.document_id
                or (chunk.metadata.source_path, chunk.metadata.source_type, chunk.metadata.title)
                != (first.source_path, first.source_type, first.title)
            ):
                raise ValueError("Chunk identity or content is inconsistent within the document")
        payload = {
            "document": document.model_dump(mode="json"),
            "model": self._provider.model, "dimensions": self._provider.dimensions,
        }
        fingerprint = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        current = await self._repository.find_current(
            first.source_path, content_hash, fingerprint, self._provider.model, len(document.chunks),
        )
        if current is not None:
            return IndexResult(current, EmbeddingUsage())
        embeddings = await self._provider.embed_documents([chunk.content for chunk in document.chunks])
        if embeddings.model != self._provider.model:
            raise ValueError("Embedding model differs from the configured provider")
        stored = await self._repository.save(document, embeddings, content_hash=content_hash, fingerprint=fingerprint)
        return IndexResult(stored, embeddings.usage)
