"""Format-independent parser output and source provenance."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class LineRange(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def check_order(self) -> "LineRange":
        if self.end < self.start:
            raise ValueError("Line range end must be at or after its start")
        return self


class ParsedSection(BaseModel):
    content: str = Field(min_length=1)
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    # Coordinates in the actual input file, including front matter lines.
    line_range: LineRange | None = None
    # May be disjoint when FAQ lines were removed from the original.
    source_line_ranges: list[LineRange] = Field(default_factory=list)


class ParseWarning(BaseModel):
    code: Literal["page_without_text"]
    page_number: int = Field(ge=1)
    message: str


class ParsedDocument(BaseModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str
    source_type: Literal["pdf", "markdown", "txt"]
    sections: list[ParsedSection] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ParseWarning] = Field(default_factory=list)


class ChunkingConfig(BaseModel):
    model_config = {"frozen": True}

    chunk_size: int = Field(default=800, gt=0, strict=True)
    chunk_overlap: int = Field(default=120, ge=0, strict=True)
    unit: Literal["characters"] = "characters"
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", ". ", "；", "，", " ", "")

    @model_validator(mode="after")
    def check_splitting_rules(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.separators or self.separators[-1] != "" or "" in self.separators[:-1]:
            raise ValueError("separators must end with a single empty-string character fallback")
        return self


class ChunkMetadata(BaseModel):
    document_id: str
    title: str
    source_path: str
    source_type: Literal["pdf", "markdown", "txt"]
    section_index: int = Field(ge=0)
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = Field(default=None, ge=1)
    # Zero-based, end-exclusive Python character offsets inside one section.
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    line_range: LineRange | None = None
    source_line_ranges: list[LineRange] = Field(default_factory=list)
    # Namespaced so source metadata cannot overwrite calculated coordinates.
    document_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_character_order(self) -> "ChunkMetadata":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class Chunk(BaseModel):
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    metadata: ChunkMetadata


class ChunkedDocument(BaseModel):
    document_id: str
    chunking_config: ChunkingConfig
    chunks: list[Chunk]
    warnings: list[ParseWarning] = Field(default_factory=list)
