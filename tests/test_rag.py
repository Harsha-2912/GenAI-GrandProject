import pytest
from pathlib import Path
from src.parser import DocumentParser
from src.chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.rag_engine import RAGEngine
from src.sample_data import generate_sample_documents
from config import SAMPLE_DOCS_DIR

@pytest.fixture(scope="module")
def rag_system():
    generate_sample_documents(SAMPLE_DOCS_DIR)
    embedding_mgr = EmbeddingManager(provider="Offline / TF-IDF Vectorizer")
    vector_store = FaissVectorStore(dimension=384)
    chunker = TextChunker()

    all_chunks = []
    # Parse and index sample files
    for f in SAMPLE_DOCS_DIR.iterdir():
        if f.suffix in [".pdf", ".txt", ".docx", ".xlsx", ".pptx"]:
            sections = DocumentParser.parse_file(f, f.name)
            chunks = chunker.chunk_sections(sections)
            all_chunks.extend(chunks)

    embeddings = embedding_mgr.embed_documents([c.text for c in all_chunks])
    vector_store.add_chunks(all_chunks, embeddings)

    engine = RAGEngine(
        vector_store=vector_store,
        embedding_manager=embedding_mgr,
        llm_provider="Local Extractive Synthesizer (No API Key Required)",
        similarity_threshold=0.20
    )
    return engine

def test_answerable_attendance(rag_system):
    q = "What is the minimum attendance requirement for students?"
    res = rag_system.answer_question(q)
    assert res.found_in_docs is True
    assert len(res.retrieved_chunks) > 0
    assert len(res.citations) > 0
    assert "75%" in res.answer or any("75%" in c.text for c, _ in res.retrieved_chunks)

def test_unanswerable_rejection(rag_system):
    # Query completely unrelated to academic documents
    q = "What is the policy regarding international vacation airfare reimbursement?"
    res = rag_system.answer_question(q, strict_threshold=0.35)
    assert res.found_in_docs is False
    assert "could not find" in res.answer.lower()
