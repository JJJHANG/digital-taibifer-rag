"""Recursive character splitting with exact section and source provenance."""

from bisect import bisect_right
from copy import deepcopy
from itertools import accumulate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.schemas import (
    Chunk,
    ChunkedDocument,
    ChunkingConfig,
    ChunkMetadata,
    LineRange,
    ParsedDocument,
    ParsedSection,
)


def _slice_source_ranges(ranges: list[LineRange], first: int, last: int) -> list[LineRange]:
    """Clip original-source segments to an inclusive section-relative line span."""
    selected: list[LineRange] = []
    offset = 0
    for source in ranges:
        width = source.end - source.start + 1
        left, right = max(first, offset), min(last, offset + width - 1)
        if left <= right:
            selected.append(LineRange(
                start=source.start + left - offset,
                end=source.start + right - offset,
            ))
        offset += width
    return selected


class RecursiveChunker:
    """Own the splitter settings, boundaries, and metadata for one document.

    Each parsed section is split independently: no overlap crosses a heading
    section or PDF page. Overlap is a target, not a guaranteed exact length.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=list(self.config.separators),
            length_function=len,
            keep_separator="end",
            strip_whitespace=False,
            add_start_index=True,
            is_separator_regex=False,
        )

    def split_document(self, document: ParsedDocument) -> ChunkedDocument:
        chunks: list[Chunk] = []
        for section_index, section in enumerate(document.sections):
            lines = section.content.splitlines(keepends=True)
            line_starts = [0, *accumulate(len(line) for line in lines[:-1])]
            self._validate_line_counts(section, len(lines))
            for split in self._splitter.create_documents([section.content]):
                content = split.page_content
                if not content.strip():
                    continue
                start = split.metadata["start_index"]
                end = start + len(content)
                if start < 0 or section.content[start:end] != content:
                    raise ValueError("Splitter returned an invalid source offset; refusing inaccurate provenance")
                first = bisect_right(line_starts, start) - 1
                last = bisect_right(line_starts, end - 1) - 1
                line_range = None
                if section.line_range is not None:
                    line_range = LineRange(
                        start=section.line_range.start + first,
                        end=section.line_range.start + last,
                    )
                chunks.append(Chunk(
                    chunk_index=len(chunks),
                    content=content,
                    metadata=ChunkMetadata(
                        document_id=document.document_id,
                        title=document.title,
                        source_path=document.source_path,
                        source_type=document.source_type,
                        section_index=section_index,
                        heading_path=list(section.heading_path),
                        page_number=section.page_number,
                        char_start=start,
                        char_end=end,
                        line_range=line_range,
                        source_line_ranges=_slice_source_ranges(section.source_line_ranges, first, last),
                        document_metadata=deepcopy(document.metadata),
                    ),
                ))
        if not chunks:
            raise ValueError("Document has no non-whitespace chunks")
        return ChunkedDocument(
            document_id=document.document_id,
            chunking_config=self.config,
            chunks=chunks,
            warnings=[warning.model_copy(deep=True) for warning in document.warnings],
        )

    @staticmethod
    def _validate_line_counts(section: ParsedSection, count: int) -> None:
        """Fail explicitly when supplied line mappings cannot match the text."""
        if section.line_range is not None:
            if section.line_range.end - section.line_range.start + 1 != count:
                raise ValueError("Section line_range does not match its content")
        if section.source_line_ranges:
            source_count = sum(span.end - span.start + 1 for span in section.source_line_ranges)
            if source_count != count:
                raise ValueError("Section source_line_ranges do not match its content")
