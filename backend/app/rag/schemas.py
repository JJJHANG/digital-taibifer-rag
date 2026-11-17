"""Retrieval and prompt contracts, independent of ORM and model SDKs."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.ingestion.embedding import EmbeddingUsage
from app.ingestion.schemas import LineRange


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, examples=["TaiBIF 與 GBIF 有什麼關係？"])
    top_k: int = Field(default=5, ge=1, le=100, strict=True)


class RetrievedChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    version_number: int
    chunk_index: int
    title: str
    source_path: str
    source_type: Literal["pdf", "markdown", "txt"]
    content: str
    distance: float = Field(description="Cosine distance; smaller values are nearer")
    score: float = Field(description="Cosine similarity (1 - distance), not confidence or probability")
    metadata: dict[str, Any]


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    embedding_model: str
    results: list[RetrievedChunk]
    usage: EmbeddingUsage


class PromptConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_context_tokens: int = Field(default=6000, gt=0, strict=True)
    max_input_tokens: int = Field(default=8000, gt=0, strict=True)
    max_question_tokens: int = Field(default=1000, gt=0, strict=True)
    max_sources: int = Field(default=10, ge=1, le=100, strict=True)
    tokenizer: Literal["cl100k_base", "o200k_base"] = "cl100k_base"


class PromptMessage(BaseModel):
    role: Literal["system", "user"]
    content: str


class SkippedContext(BaseModel):
    chunk_id: UUID
    reason: Literal["duplicate", "empty_content", "source_limit", "context_budget", "input_budget"]


class PreparedPrompt(BaseModel):
    prompt_version: str
    messages: list[PromptMessage]
    selected_chunk_ids: list[UUID]
    skipped_context: list[SkippedContext]
    estimated_context_tokens: int
    estimated_input_tokens: int
    tokenizer: Literal["cl100k_base", "o200k_base"] = "cl100k_base"
    config: PromptConfig


class ChatUsage(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    requests: int = Field(default=1, ge=0)


class GenerationResult(BaseModel):
    text: str = Field(min_length=1)
    status: Literal["completed", "refused"]
    requested_model: str
    model: str
    response_id: str
    usage: ChatUsage


class CitationSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    version_number: int
    chunk_index: int
    title: str
    source_path: str
    source_type: Literal["pdf", "markdown", "txt"]
    source_url: HttpUrl | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    source_line_ranges: list[LineRange] = Field(default_factory=list)


class CitationReport(BaseModel):
    status: Literal["valid", "invalid", "missing", "not_applicable"] = "not_applicable"
    cited_ids: list[str] = Field(default_factory=list)
    invalid_markers: list[str] = Field(default_factory=list)
    sources: list[CitationSource] = Field(default_factory=list)
    grounding_checked: Literal[False] = False


class RAGResult(BaseModel):
    """Internal diagnostics, not the future public /chat response schema."""

    query: str
    answer: str
    status: Literal["generated", "model_refused", "insufficient_context"]
    retrieval: RetrievalResult
    prompt: PreparedPrompt
    generation: GenerationResult | None
    citations: CitationReport = Field(default_factory=CitationReport)
