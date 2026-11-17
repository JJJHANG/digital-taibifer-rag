"""Public chat contracts: answer, source identities and bounded diagnostics."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.embedding import EmbeddingUsage
from app.rag.schemas import ChatUsage, CitationSource


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(min_length=1, examples=["TaiBIF 與 GBIF 有什麼關係？"])
    top_k: int = Field(default=5, ge=1, le=100, strict=True)


class ChatCitationStatus(BaseModel):
    status: Literal["valid", "missing", "not_applicable"]
    cited_ids: list[str]
    grounding_checked: Literal[False] = False


class ChatRetrievalHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    score: float
    included_in_context: bool


class ChatRetrievalSummary(BaseModel):
    top_k: int
    retrieved_count: int
    context_count: int
    skipped_context_count: int
    embedding_model: str
    chunks: list[ChatRetrievalHit]


class ChatUsageSummary(BaseModel):
    embedding: EmbeddingUsage
    generation: ChatUsage | None


class ChatResponse(BaseModel):
    message: str
    answer: str
    status: Literal["generated", "model_refused", "insufficient_context"]
    sources: list[CitationSource]
    citations: ChatCitationStatus
    retrieval: ChatRetrievalSummary
    chat_model: str | None
    prompt_version: str
    usage: ChatUsageSummary
