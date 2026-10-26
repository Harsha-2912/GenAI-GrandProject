import os
import io
import time
import json
from pathlib import Path
import streamlit as st
import pandas as pd

# Core DocuSage Modules
from config import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    GEMINI_MODELS,
    OPENAI_MODELS,
    GROQ_MODELS,
    SAMPLE_DOCS_DIR,
    INDEX_DIR,
    INSTITUTIONAL_DISCLAIMER
)
from src.parser import DocumentParser
from src.chunker import TextChunker, DocumentChunk
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.rag_engine import RAGEngine, RAGResponse
from src.sample_data import generate_sample_documents
from src.evaluator import RAGEvaluator
from src.utils import (
    get_file_icon,
    human_readable_size,
    export_conversation_to_markdown,
    export_conversation_to_json
)

# Set page configuration
st.set_page_config(
    page_title="DocuSage AI — Document Intelligence RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling (Dark Modern Glassmorphism)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 50%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* Warning / Institutional Banner */
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #F59E0B;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.84rem;
        color: #FCD34D;
        margin-bottom: 1.2rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Answer Container */
    .answer-box {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 14px;
        margin-bottom: 16px;
    }

    /* Citation Pill */
    .citation-pill {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 3px 4px 3px 0;
    }

    /* Evidence Passage Box */
    .passage-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
    .passage-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        padding-bottom: 4px;
    }
    .relevance-score {
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "embedding_provider" not in st.session_state:
    st.session_state.embedding_provider = EMBEDDING_PROVIDERS[0]

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = LLM_PROVIDERS[3]  # Default: Local Extractive Synthesizer

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-1.5-flash"

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD

if "top_k" not in st.session_state:
    st.session_state.top_k = DEFAULT_TOP_K

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_doc_filter" not in st.session_state:
    st.session_state.selected_doc_filter = []

@st.cache_resource(show_spinner="Loading Embedding Model in memory...")
def load_embedding_manager(provider_name: str, api_key_val: str):
    return EmbeddingManager(provider=provider_name, api_key=api_key_val)

embedding_mgr = load_embedding_manager(st.session_state.embedding_provider, st.session_state.api_key)

if "vector_store" not in st.session_state:
    store = FaissVectorStore(dimension=384)
    # Check if index exists on disk to load automatically
    if (INDEX_DIR / "index.faiss").exists():
        store.load_from_disk(INDEX_DIR)
    st.session_state.vector_store = store

vector_store: FaissVectorStore = st.session_state.vector_store

# Instantiate RAG Engine
rag_engine = RAGEngine(
    vector_store=vector_store,
    embedding_manager=embedding_mgr,
    llm_provider=st.session_state.llm_provider,
    model_name=st.session_state.selected_model,
    api_key=st.session_state.api_key,
    similarity_threshold=st.session_state.similarity_threshold
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/books.png", width=56)
    st.markdown("### 🎓 **DocuSage AI Control Center**")
    st.caption("Institutional Document Intelligence & Grounded RAG")

    st.markdown("---")
    st.markdown("#### 🤖 **LLM & Inference Engine**")
    selected_llm = st.selectbox(
        "LLM Provider",
        LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state.llm_provider),
        help="Select generative model provider. 'Local Extractive Synthesizer' works 100% offline without API keys."
    )
    st.session_state.llm_provider = selected_llm

    if selected_llm == "Google Gemini":
        st.session_state.selected_model = st.selectbox("Gemini Model", GEMINI_MODELS)
        api_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password", help="Enter your Google Gemini API Key")
        if api_input:
            st.session_state.api_key = api_input
    elif selected_llm == "OpenAI":
        st.session_state.selected_model = st.selectbox("OpenAI Model", OPENAI_MODELS)
        api_input = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
        if api_input:
            st.session_state.api_key = api_input
    elif selected_llm == "Groq":
        st.session_state.selected_model = st.selectbox("Groq Model", GROQ_MODELS)
        api_input = st.text_input("Groq API Key", value=st.session_state.api_key, type="password")
        if api_input:
            st.session_state.api_key = api_input
    else:
        st.info("⚡ Zero-API Mode: Using local extractive synthesizer grounded strictly in retrieved chunks.")

    st.markdown("---")
    st.markdown("#### 🎯 **Retrieval Hyperparameters**")
    top_k_val = st.slider("Top-K Passages", min_value=1, max_value=8, value=st.session_state.top_k)
    st.session_state.top_k = top_k_val

    threshold_val = st.slider(
        "Min Similarity Threshold",
        min_value=0.10,
        max_value=0.80,
        value=st.session_state.similarity_threshold,
        step=0.02,
        help="Questions whose retrieved passages score below this threshold will be classified as unanswerable / insufficient evidence."
    )
    st.session_state.similarity_threshold = threshold_val

    st.markdown("---")
    # Quick Sample Documents Loader
    st.markdown("#### 🚀 **Quick Start Demo**")
    if st.button("⚡ Load 5 Sample College Documents", use_container_width=True):
        with st.spinner("Generating and indexing 5 approved institutional documents (PDF, TXT, Excel, PPTX, DOCX)..."):
            generate_sample_documents(SAMPLE_DOCS_DIR)
            chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
            total_added = 0
            for sample_file in SAMPLE_DOCS_DIR.iterdir():
                if sample_file.suffix.lower() in ALLOWED_EXTENSIONS and sample_file.name not in vector_store.document_registry:
                    sections = DocumentParser.parse_file(sample_file, sample_file.name)
                    chunks = chunker.chunk_sections(sections)
                    if chunks:
                        embs = embedding_mgr.embed_documents([c.text for c in chunks])
                        vector_store.add_chunks(chunks, embs)
                        total_added += len(chunks)
            if total_added > 0:
                vector_store.save_to_disk(INDEX_DIR)
                st.success(f"Successfully indexed 5 sample documents ({total_added} chunks)!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.info("Sample documents are already present in the index.")

    st.markdown("---")
    st.caption("DocuSage AI v1.0.0 • Built for Institutional Grounding")


# ---------------------------------------------------------
# Main App Header & Metrics
# ---------------------------------------------------------
st.markdown(f"""
<div class="brand-container">
    <div>
        <h1 class="brand-title">DocuSage AI</h1>
        <p class="brand-subtitle">Document Intelligence & Retrieval-Augmented Generation System</p>
    </div>
    <div style="text-align: right;">
        <span class="status-badge">● Active Knowledge Base</span>
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 5px;">Model: {st.session_state.llm_provider}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Institutional Disclaimer Banner
st.markdown(f"""
<div class="disclaimer-banner">
    ⚠️ <strong>Institutional Compliance Notice:</strong> {INSTITUTIONAL_DISCLAIMER}
</div>
""", unsafe_allow_html=True)

# Live Statistics Metric Badges
indexed_summary = vector_store.get_indexed_files_summary()
total_docs = len(indexed_summary)
total_chunks = len(vector_store.chunks)
total_chars = sum(doc.get("Characters", 0) for doc in indexed_summary)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_docs}</div><div class="metric-label">Approved Documents</div></div>""", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_chunks}</div><div class="metric-label">Indexed Chunks</div></div>""", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{total_chars:,}</div><div class="metric-label">Characters Parsed</div></div>""", unsafe_allow_html=True)
with col_m4:
    threshold_pct = f"{st.session_state.similarity_threshold:.0%}"
    st.markdown(f"""<div class="metric-card"><div class="metric-value">{threshold_pct}</div><div class="metric-label">Min Relevance Gate</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_qa, tab_index, tab_eval, tab_help = st.tabs([
    "💬 Ask DocuSage (Q&A)",
    "📁 Document Index & Ingestion",
    "🧪 Benchmark Evaluation Suite",
    "ℹ️ Architecture & Guidelines"
])

# =========================================================
# TAB 1: Q&A & Document Intelligence
# =========================================================
with tab_qa:
    if total_docs == 0:
        st.warning("No documents are currently indexed in the system. Please upload files in the 'Document Index & Ingestion' tab or click '⚡ Load 5 Sample College Documents' in the sidebar!")
    else:
        # Document Filter Row (Advanced Requirement)
        doc_names = list(vector_store.document_registry.keys())
        c_filter, c_reset = st.columns([4, 1])
        with c_filter:
            selected_docs = st.multiselect(
                "Filter Retrieval by Document (Optional):",
                options=doc_names,
                default=[],
                help="Leave empty to search across all approved documents, or select specific files to constrain retrieval."
            )
        with c_reset:
            st.write("")
            st.write("")
            if selected_docs and st.button("Clear Filter"):
                st.session_state.selected_doc_filter = []
                st.rerun()

        filter_target = selected_docs if selected_docs else None

        # Sample Quick Questions Chips
        st.markdown("**Sample institutional queries to test groundedness & rejection:**")
        sample_q_cols = st.columns(4)
        sample_queries = [
            "What is the minimum attendance requirement for semester exams?",
            "When is the fee clearance deadline for Fall 2024?",
            "What are the central library operating hours on weekends?",
            "What is the airfare reimbursement policy for vacations? (Unanswerable)"
        ]

        clicked_query = None
        for i, sq in enumerate(sample_queries):
            with sample_queries_col := sample_q_cols[i]:
                if st.button(sq, key=f"sq_btn_{i}", use_container_width=True):
                    clicked_query = sq

        # Query Input
        user_query = st.text_input(
            "Ask a question grounded in the approved documents:",
            value=clicked_query if clicked_query else "",
            placeholder="e.g. What is the fee for semester exam re-evaluation?"
        )

        ask_btn = st.button("🔍 Search & Answer", type="primary", use_container_width=True)

        if ask_btn and user_query.strip():
            with st.spinner("Retrieving relevant passages and synthesizing grounded answer..."):
                start_time = time.time()
                response = rag_engine.answer_question(
                    question=user_query,
                    top_k=st.session_state.top_k,
                    filter_filenames=filter_target,
                    strict_threshold=st.session_state.similarity_threshold
                )
                latency = time.time() - start_time

                # Save to session history
                st.session_state.chat_history.insert(0, {
                    "question": user_query,
                    "answer": response.answer,
                    "found_in_docs": response.found_in_docs,
                    "citations": response.citations,
                    "retrieved_chunks": response.retrieved_chunks,
                    "model": response.model_used,
                    "latency": f"{latency:.2f}s"
                })

        # Display latest or current Q&A in Dual Column / Side-by-Side View
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[0]
            
            st.markdown("---")
            st.markdown(f"### 📋 **Question:** {latest['question']}")

            col_ans, col_evidence = st.columns([3, 2])

            with col_ans:
                # Answer box with groundedness indicators
                grounded_color = "#4ADE80" if latest["found_in_docs"] else "#F59E0B"
                grounded_label = "✅ Grounded in Approved Evidence" if latest["found_in_docs"] else "⚠️ Insufficient Evidence / Not Found"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: {grounded_color}; font-weight: 600; font-size: 0.88rem;">{grounded_label}</span>
                    <span style="color: #94A3B8; font-size: 0.78rem;">Model: {latest['model']} ({latest.get('latency', '')})</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="answer-box">
                    {latest['answer']}
                </div>
                """, unsafe_allow_html=True)

                # Show Verified Citations Pills
                if latest["citations"]:
                    st.markdown("**Verified Source Citations:**")
                    citation_html = ""
                    for cit in latest["citations"]:
                        citation_html += f"""<span class="citation-pill">{cit.tag} (Match: {cit.similarity_score:.1%})</span>"""
                    st.markdown(citation_html, unsafe_allow_html=True)

            with col_evidence:
                st.markdown("### 🔍 **Retrieved Evidence Passages**")
                st.caption(f"Showing Top-{len(latest['retrieved_chunks'])} passages fetched from FAISS:")
                
                if not latest["retrieved_chunks"]:
                    st.info("No passages met the minimum vector retrieval similarity.")
                else:
                    for idx, (chunk, score) in enumerate(latest["retrieved_chunks"], start=1):
                        icon = get_file_icon(chunk.file_type)
                        st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>{icon} {chunk.filename} — {chunk.section_label}</span>
                                <span class="relevance-score">Score: {score:.1%}</span>
                            </div>
                            <div style="color: #E2E8F0; line-height: 1.45;">
                                {chunk.text}
                            </div>
                            <div style="margin-top: 6px; font-size: 0.72rem; color: #64748B;">
                                Chunk ID: <code>{chunk.chunk_id}</code> | {chunk.char_count} chars
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # Export Q&A History
            st.markdown("---")
            exp_c1, exp_c2, exp_c3 = st.columns([2, 1, 1])
            with exp_c1:
                st.caption(f"Q&A Session History: {len(st.session_state.chat_history)} queries recorded.")
            with exp_c2:
                md_data = export_conversation_to_markdown(st.session_state.chat_history)
                st.download_button(
                    label="📥 Export Report (Markdown)",
                    data=md_data,
                    file_name="docusage_qa_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with exp_c3:
                json_data = export_conversation_to_json(st.session_state.chat_history)
                st.download_button(
                    label="📥 Export JSON",
                    data=json_data,
                    file_name="docusage_qa_history.json",
                    mime="application/json",
                    use_container_width=True
                )

# =========================================================
# TAB 2: Document Index & Ingestion
# =========================================================
with tab_index:
    st.markdown("### 📁 **Document Ingestion & Index Management**")
    st.write(
        "Upload approved institutional documents (handbooks, examination notices, syllabi, matrices). "
        "DocuSage automatically parses each format, generates embeddings, and indexes them into FAISS."
    )

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, TXT, Excel, PPTX, DOCX):",
        type=["pdf", "txt", "xlsx", "xls", "csv", "pptx", "docx"],
        accept_multiple_files=True,
        help="Max 25 MB per file. Institutional authorization required."
    )

    if uploaded_files:
        upload_btn = st.button("⚡ Process & Index Uploaded Documents", type="primary")
        if upload_btn:
            chunker = TextChunker(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            total_files = len(uploaded_files)

            new_chunks_count = 0
            for i, u_file in enumerate(uploaded_files):
                status_text.text(f"Parsing and chunking: {u_file.name} ({i+1}/{total_files})...")
                # Parse
                file_bytes = io.BytesIO(u_file.read())
                sections = DocumentParser.parse_file(file_bytes, u_file.name)
                chunks = chunker.chunk_sections(sections)

                if chunks:
                    status_text.text(f"Generating embeddings for {len(chunks)} chunks in {u_file.name}...")
                    embs = embedding_mgr.embed_documents([c.text for c in chunks])
                    vector_store.add_chunks(chunks, embs)
                    new_chunks_count += len(chunks)

                progress_bar.progress((i + 1) / total_files)

            # Persist index
            vector_store.save_to_disk(INDEX_DIR)
            status_text.empty()
            progress_bar.empty()
            st.success(f"Successfully processed {len(uploaded_files)} files ({new_chunks_count} chunks indexed)!")
            time.sleep(0.5)
            st.rerun()

    st.markdown("---")
    st.markdown("#### 📑 **Currently Indexed Approved Documents**")

    if not indexed_summary:
        st.info("No documents are currently indexed.")
    else:
        df_summary = pd.DataFrame(indexed_summary)
        st.dataframe(df_summary, use_container_width=True)

        st.markdown("#### ⚙️ **Document Actions & Removal**")
        doc_col1, doc_col2, doc_col3 = st.columns([2, 1, 1])

        with doc_col1:
            doc_to_delete = st.selectbox("Select Document to Remove:", options=list(vector_store.document_registry.keys()))
            if st.button("🗑️ Delete Selected Document", type="secondary"):
                deleted = vector_store.delete_document(doc_to_delete, embedding_manager=embedding_mgr)
                if deleted:
                    vector_store.save_to_disk(INDEX_DIR)
                    st.success(f"Document '{doc_to_delete}' removed from index.")
                    time.sleep(0.5)
                    st.rerun()

        with doc_col2:
            st.write("")
            st.write("")
            if st.button("🔄 Refresh FAISS Index", use_container_width=True):
                vector_store.save_to_disk(INDEX_DIR)
                st.success("Vector store index refreshed and persisted.")

        with doc_col3:
            st.write("")
            st.write("")
            if st.button("⚠️ Clear Entire Index", use_container_width=True):
                vector_store.clear()
                # Remove disk files
                for f in [INDEX_DIR / "index.faiss", INDEX_DIR / "chunks.json", INDEX_DIR / "registry.json"]:
                    if f.exists():
                        f.unlink()
                st.warning("All indexed documents and vectors cleared.")
                time.sleep(0.5)
                st.rerun()

        # Raw Chunk Inspector (Advanced Extension)
        with st.expander("🔍 Inspect Extracted Chunks & Citation Lineage"):
            inspect_doc = st.selectbox("Select Document to Inspect:", options=list(vector_store.document_registry.keys()), key="inspect_doc_select")
            doc_chunks = [c for c in vector_store.chunks if c.filename == inspect_doc]
            st.caption(f"Showing {len(doc_chunks)} chunks for '{inspect_doc}':")
            for c in doc_chunks[:10]:
                st.markdown(f"**Chunk ID**: `{c.chunk_id}` | **Citation**: `{c.citation_tag}` | **Size**: `{c.char_count} chars`")
                st.text(c.text)
                st.markdown("---")

# =========================================================
# TAB 3: Benchmark & Evaluation Suite
# =========================================================
with tab_eval:
    st.markdown("### 🧪 **RAG Groundedness & Retrieval Benchmark Suite**")
    st.write(
        "Evaluate retrieval precision, citation accuracy, and rejection of out-of-domain / unanswerable questions "
        "against an institutional ground-truth benchmark."
    )

    evaluator = RAGEvaluator(rag_engine=rag_engine)
    benchmark_file = Path("evaluation_benchmark.json")

    if not benchmark_file.exists():
        st.error("evaluation_benchmark.json was not found in the workspace.")
    else:
        test_cases = evaluator.load_benchmark(benchmark_file)
        st.info(f"Loaded benchmark dataset containing **{len(test_cases)}** institutional test cases (Answerable + Unanswerable).")

        run_eval_btn = st.button("▶️ Run Complete Evaluation Suite", type="primary")

        if run_eval_btn:
            eval_progress = st.progress(0.0)
            eval_status = st.empty()

            def progress_cb(current, total, query):
                eval_progress.progress(current / total)
                eval_status.text(f"Evaluating ({current}/{total}): {query[:60]}...")

            summary = evaluator.run_benchmark(test_cases, progress_callback=progress_cb)
            eval_progress.empty()
            eval_status.empty()

            st.session_state.eval_summary = summary

        if "eval_summary" in st.session_state:
            summary = st.session_state.eval_summary

            # Visual Scorecard
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{summary.accuracy_percentage:.1f}%</div><div class="metric-label">Overall Accuracy</div></div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{summary.retrieval_hit_rate:.1f}%</div><div class="metric-label">Retrieval Hit Rate</div></div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{summary.unanswerable_rejection_rate:.1f}%</div><div class="metric-label">Unanswerable Rejection</div></div>""", unsafe_allow_html=True)
            with sc4:
                st.markdown(f"""<div class="metric-card"><div class="metric-value">{summary.answerable_accuracy:.1f}%</div><div class="metric-label">Answerable Grounding</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Results Breakdown Table
            rows = []
            for r in summary.results:
                status_icon = "✅ PASS" if r.verdict == "PASS" else "❌ FAIL"
                rows.append({
                    "ID": r.test_id,
                    "Category": r.category,
                    "Question": r.question,
                    "Type": "Answerable" if r.is_answerable else "Unanswerable",
                    "Expected Doc": r.expected_document or "None (Reject)",
                    "Retrieved Docs": ", ".join(r.retrieved_documents) or "None",
                    "Verdict": status_icon,
                    "Explanation": r.explanation
                })

            st.markdown("#### 📊 **Detailed Test Case Results**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Expandable Full Answers Inspector
            with st.expander("🔍 View Full Generated Answers for Each Test Case"):
                for r in summary.results:
                    st.markdown(f"**[{r.verdict}] {r.test_id}: {r.question}**")
                    st.markdown(f"*Generated Response:*")
                    st.info(r.answer_generated)
                    st.markdown("---")

# =========================================================
# TAB 4: Architecture & Guidelines
# =========================================================
with tab_help:
    st.markdown("### 🏛️ **DocuSage AI Architecture & Operational Guide**")
    st.markdown("""
    #### 1. System Pipeline
    DocuSage AI operates on a **4-stage decoupled architecture**:
    - **Document Ingestion & Multi-Format Parsing**: Extracts text from PDFs (PyMuPDF/PyPDF), Excel workbooks (OpenPyXL), PowerPoint decks (Python-PPTX), Word documents (Python-Docx), and raw Text files. Preserves exact page, slide, and sheet metadata.
    - **Intelligent Recursive Chunking**: Segments sections into overlapping chunks (600 characters, 100 character overlap) while preserving sentence integrity and citation lineage tags (e.g. `[Handbook.pdf, Page 2]`).
    - **Vector Search & Grounding Threshold**: Projects chunks into normalized embedding space using FAISS (`IndexFlatIP`). Evaluates similarity against a configurable threshold to catch unanswerable questions before LLM synthesis.
    - **Strict Grounded Synthesis**: Passes retrieved context to the LLM with strict negative constraints (no external extrapolation) and enforces document & page citation brackets.
    """)

    st.markdown("""
    #### 2. Institutional Privacy & Confidentiality Policy
    - Never upload confidential student personally identifiable information (PII), payroll records, or sensitive disciplinary records without explicit written institutional consent.
    - All vector indices and parsed chunks remain in your private workspace directory.
    """)

    st.markdown("""
    #### 3. Deployment Checklist (Streamlit Community Cloud)
    1. Push this project repository to GitHub.
    2. Go to [share.streamlit.io](https://share.streamlit.io) and select your repo.
    3. Specify `app.py` as the main entry file.
    4. Under Advanced Settings, add your API keys (e.g. `GEMINI_API_KEY`, `OPENAI_API_KEY`) as Secrets if using API-based models.
    5. Click Deploy!
    """)
