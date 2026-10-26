import pytest
from src.parser import DocumentSection
from src.chunker import TextChunker, DocumentChunk

def test_chunker_basic():
    long_text = (
        "University attendance policy requires 75% minimum presence. "
        "Students who fail to meet this threshold will be detained from semester examinations. "
        "A condonation fee of 500 rupees applies for genuine medical reasons. "
        "Medical certificates must be submitted within 5 days."
    )
    section = DocumentSection(
        text=long_text,
        filename="Handbook.pdf",
        file_type="pdf",
        section_type="page",
        section_index=2,
        section_label="Page 2",
        metadata={"page_number": 2}
    )

    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_sections([section])

    assert len(chunks) >= 2, "Long section should split into multiple chunks"
    for c in chunks:
        assert c.filename == "Handbook.pdf"
        assert c.citation_tag == "[Handbook.pdf, Page 2]"
        assert len(c.text) <= 150

def test_chunker_overlap():
    chunker = TextChunker(chunk_size=50, chunk_overlap=20)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=50, chunk_overlap=50)

def test_chunker_metadata():
    section = DocumentSection(
        text="Library is open from 8am to 10pm on weekdays.",
        filename="Guide.pptx",
        file_type="pptx",
        section_type="slide",
        section_index=3,
        section_label="Slide 3",
        metadata={"slide_number": 3}
    )
    chunker = TextChunker()
    chunks = chunker.chunk_sections([section])
    assert len(chunks) == 1
    assert chunks[0].citation_tag == "[Guide.pptx, Slide 3]"
    assert chunks[0].word_count > 0
