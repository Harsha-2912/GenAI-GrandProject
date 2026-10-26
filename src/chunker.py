import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.parser import DocumentSection

@dataclass
class DocumentChunk:
    """Represents an embedded search chunk with complete lineage & citation metadata."""
    chunk_id: str
    text: str
    filename: str
    file_type: str
    section_type: str  # 'page', 'slide', 'sheet', 'section', 'document'
    section_index: int  # 1-indexed (e.g. Page 1, Slide 3, Sheet 2)
    section_label: str  # e.g. "Page 4", "Slide 2: Library Rules"
    chunk_index: int
    char_count: int
    word_count: int
    citation_tag: str  # e.g. "[Academic_Handbook.pdf, Page 4]"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "filename": self.filename,
            "file_type": self.file_type,
            "section_type": self.section_type,
            "section_index": self.section_index,
            "section_label": self.section_label,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "citation_tag": self.citation_tag,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class TextChunker:
    """Intelligent recursive text chunker that preserves document structure, sentences, and metadata."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_sections(self, sections: List[DocumentSection]) -> List[DocumentChunk]:
        """Splits a list of DocumentSections into a list of DocumentChunks."""
        all_chunks: List[DocumentChunk] = []

        for section in sections:
            section_chunks = self._chunk_single_section(section)
            all_chunks.extend(section_chunks)

        return all_chunks

    def _chunk_single_section(self, section: DocumentSection) -> List[DocumentChunk]:
        raw_text = section.text.strip()
        if not raw_text:
            return []

        # If section is already within chunk size, keep as single chunk
        if len(raw_text) <= self.chunk_size:
            text_blocks = [raw_text]
        else:
            text_blocks = self._recursive_split(raw_text, self.chunk_size, self.chunk_overlap)

        chunks: List[DocumentChunk] = []
        doc_hash = hashlib.md5(section.filename.encode("utf-8")).hexdigest()[:8]

        for idx, block in enumerate(text_blocks, start=1):
            cleaned_block = block.strip()
            if not cleaned_block:
                continue

            chunk_id = f"{doc_hash}_{section.section_type}{section.section_index}_c{idx}"
            
            # Format appropriate citation reference tag
            if section.section_type == "page":
                citation_tag = f"[{section.filename}, Page {section.section_index}]"
            elif section.section_type == "slide":
                citation_tag = f"[{section.filename}, Slide {section.section_index}]"
            elif section.section_type == "sheet":
                sheet_name = section.metadata.get("sheet_name", f"Sheet {section.section_index}")
                citation_tag = f"[{section.filename}, Sheet: {sheet_name}]"
            else:
                citation_tag = f"[{section.filename}, Ref {section.section_index}]"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=cleaned_block,
                filename=section.filename,
                file_type=section.file_type,
                section_type=section.section_type,
                section_index=section.section_index,
                section_label=section.section_label,
                chunk_index=idx,
                char_count=len(cleaned_block),
                word_count=len(cleaned_block.split()),
                citation_tag=citation_tag,
                metadata=section.metadata
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Recursively splits text by paragraphs, newlines, sentences, and words.
        """
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        return self._split_with_separators(text, chunk_size, chunk_overlap, separators)

    def _split_with_separators(self, text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        if not separators:
            # Hard character cut if no separator is found
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start += chunk_size - chunk_overlap
            return chunks

        sep = separators[0]
        remaining_seps = separators[1:]
        splits = text.split(sep)

        result_chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for segment in splits:
            seg_len = len(segment) + (len(sep) if current_chunk else 0)

            if current_len + seg_len <= chunk_size:
                current_chunk.append(segment)
                current_len += seg_len
            else:
                if current_chunk:
                    joined = sep.join(current_chunk).strip()
                    if joined:
                        result_chunks.append(joined)
                    
                    # Compute overlap
                    overlap_items: List[str] = []
                    overlap_len = 0
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) + len(sep) <= chunk_overlap:
                            overlap_items.insert(0, item)
                            overlap_len += len(item) + len(sep)
                        else:
                            break
                    current_chunk = overlap_items
                    current_len = overlap_len

                # Handle single segment larger than chunk_size
                if len(segment) > chunk_size:
                    sub_chunks = self._split_with_separators(segment, chunk_size, chunk_overlap, remaining_seps)
                    for sc in sub_chunks[:-1]:
                        if sc.strip():
                            result_chunks.append(sc.strip())
                    if sub_chunks:
                        current_chunk = [sub_chunks[-1]]
                        current_len = len(sub_chunks[-1])
                else:
                    current_chunk.append(segment)
                    current_len += len(segment) + (len(sep) if len(current_chunk) > 1 else 0)

        if current_chunk:
            joined = sep.join(current_chunk).strip()
            if joined:
                result_chunks.append(joined)

        return result_chunks
