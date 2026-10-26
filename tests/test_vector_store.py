import pytest
import numpy as np
from src.chunker import DocumentChunk
from src.vector_store import FaissVectorStore

def test_vector_store_add_and_search():
    store = FaissVectorStore(dimension=4)
    c1 = DocumentChunk("c1", "Attendance 75%", "Policy.pdf", "pdf", "page", 1, "Page 1", 1, 15, 2, "[Policy.pdf, Page 1]")
    c2 = DocumentChunk("c2", "Library hours 8am", "Facilities.pptx", "pptx", "slide", 2, "Slide 2", 1, 18, 3, "[Facilities.pptx, Slide 2]")
    
    # 2 normalized vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.stack([v1, v2])

    store.add_chunks([c1, c2], embeddings)
    assert len(store.chunks) == 2
    assert store.index.ntotal == 2

    # Query matching v1
    q = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    q = q / np.linalg.norm(q)
    results = store.search(q, top_k=2)

    assert len(results) == 2
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0.8

def test_document_filtering():
    store = FaissVectorStore(dimension=4)
    c1 = DocumentChunk("c1", "Text 1", "DocA.pdf", "pdf", "page", 1, "Page 1", 1, 10, 2, "[DocA.pdf, Page 1]")
    c2 = DocumentChunk("c2", "Text 2", "DocB.pdf", "pdf", "page", 1, "Page 1", 1, 10, 2, "[DocB.pdf, Page 1]")
    
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)
    store.add_chunks([c1, c2], np.stack([v1, v2]))

    # Filter to only DocB.pdf
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(q, top_k=2, filter_filenames=["DocB.pdf"])
    assert len(results) == 1
    assert results[0][0].filename == "DocB.pdf"

def test_document_deletion():
    store = FaissVectorStore(dimension=4)
    c1 = DocumentChunk("c1", "Text 1", "DocA.pdf", "pdf", "page", 1, "Page 1", 1, 10, 2, "[DocA.pdf, Page 1]")
    c2 = DocumentChunk("c2", "Text 2", "DocB.pdf", "pdf", "page", 1, "Page 1", 1, 10, 2, "[DocB.pdf, Page 1]")
    
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    store.add_chunks([c1, c2], np.stack([v1, v2]))

    deleted = store.delete_document("DocA.pdf")
    assert deleted is True
    assert len(store.chunks) == 1
    assert store.chunks[0].filename == "DocB.pdf"
