"""Import mapped models so Alembic can discover every table."""

from app.models.base import Base
from app.models.documents import Document, DocumentChunk, DocumentVersion

__all__ = ["Base", "Document", "DocumentChunk", "DocumentVersion"]
