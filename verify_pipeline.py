import sys
from pathlib import Path
from src.sample_data import generate_sample_documents
from src.parser import DocumentParser
from src.chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.rag_engine import RAGEngine
from src.evaluator import RAGEvaluator

def main():
    print("=== Step 1: Generating and indexing documents ===", flush=True)
    sample_dir = Path("sample_documents")
    generate_sample_documents(sample_dir)

    chunker = TextChunker()
    # Use offline TF-IDF vectorizer for instantaneous deterministic verification
    emb_mgr = EmbeddingManager(provider="Offline / TF-IDF Vectorizer")
    v_store = FaissVectorStore(dimension=384)

    all_chunks = []
    for f in sample_dir.iterdir():
        if f.suffix in [".pdf", ".txt", ".docx", ".xlsx", ".pptx"]:
            secs = DocumentParser.parse_file(f, f.name)
            chunks = chunker.chunk_sections(secs)
            all_chunks.extend(chunks)

    print(f"Total chunks parsed: {len(all_chunks)}", flush=True)
    embs = emb_mgr.embed_documents([c.text for c in all_chunks])
    v_store.add_chunks(all_chunks, embs)
    print(f"Indexed files: {list(v_store.document_registry.keys())}", flush=True)

    engine = RAGEngine(v_store, emb_mgr, similarity_threshold=0.25)

    print("\n=== Step 2: Testing Answerable Question ===", flush=True)
    ans_res = engine.answer_question("What is the mandatory minimum attendance percentage required to sit for the end-semester examinations?")
    print(f"Question: {ans_res.question}", flush=True)
    print(f"Found in docs: {ans_res.found_in_docs}", flush=True)
    if ans_res.citations:
        print(f"Top Citation: {ans_res.citations[0].tag} (Score: {ans_res.citations[0].similarity_score:.1%})", flush=True)
    print("Answer Snippet:\n", ans_res.answer[:250], flush=True)

    print("\n=== Step 3: Testing Unanswerable Question ===", flush=True)
    unans_res = engine.answer_question("What is the flight reimbursement policy for student international vacations?", strict_threshold=0.35)
    print(f"Question: {unans_res.question}", flush=True)
    print(f"Found in docs: {unans_res.found_in_docs}", flush=True)
    print("Answer:\n", unans_res.answer[:250], flush=True)

    print("\n=== Step 4: Running Benchmark Suite ===", flush=True)
    evaluator = RAGEvaluator(engine)
    cases = evaluator.load_benchmark(Path("evaluation_benchmark.json"))
    summary = evaluator.run_benchmark(cases)
    print(f"Total Tests: {summary.total_tests}", flush=True)
    print(f"Passed: {summary.passed_tests}", flush=True)
    print(f"Failed: {summary.failed_tests}", flush=True)
    print(f"Accuracy: {summary.accuracy_percentage:.1f}%", flush=True)
    print(f"Retrieval Hit Rate: {summary.retrieval_hit_rate:.1f}%", flush=True)
    print(f"Unanswerable Rejection Rate: {summary.unanswerable_rejection_rate:.1f}%", flush=True)

if __name__ == "__main__":
    main()
