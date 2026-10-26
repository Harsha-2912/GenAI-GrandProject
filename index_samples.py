from pathlib import Path
from src.sample_data import generate_sample_documents
from src.parser import DocumentParser
from src.chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from config import INDEX_DIR, SAMPLE_DOCS_DIR

def index_samples():
    print("Generating sample documents...", flush=True)
    generate_sample_documents(SAMPLE_DOCS_DIR)

    chunker = TextChunker()
    emb_mgr = EmbeddingManager(provider="Local (SentenceTransformers: all-MiniLM-L6-v2)")
    v_store = FaissVectorStore(dimension=384)

    all_chunks = []
    for f in SAMPLE_DOCS_DIR.iterdir():
        if f.suffix in [".pdf", ".txt", ".docx", ".xlsx", ".pptx"]:
            secs = DocumentParser.parse_file(f, f.name)
            chunks = chunker.chunk_sections(secs)
            all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}", flush=True)
    embs = emb_mgr.embed_documents([c.text for c in all_chunks])
    v_store.add_chunks(all_chunks, embs)
    v_store.save_to_disk(INDEX_DIR)
    print(f"Index successfully saved to {INDEX_DIR}", flush=True)

if __name__ == "__main__":
    index_samples()
