import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import RAG_SYSTEM_PROMPT, UNANSWERABLE_PHRASE, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
from src.chunker import DocumentChunk
from src.vector_store import FaissVectorStore
from src.embeddings import EmbeddingManager

@dataclass
class Citation:
    """Detailed citation reference supporting an answer."""
    tag: str  # e.g. [Academic_Handbook.pdf, Page 4]
    filename: str
    section_label: str
    similarity_score: float
    snippet: str

@dataclass
class RAGResponse:
    """Standardized response from the DocuSage AI RAG pipeline."""
    question: str
    answer: str
    found_in_docs: bool
    retrieved_chunks: List[Tuple[DocumentChunk, float]]
    citations: List[Citation]
    model_used: str
    tokens_or_latency_info: str = ""

class RAGEngine:
    """Core RAG engine managing retrieval, relevance evaluation, LLM synthesis, and citation verification."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_manager: EmbeddingManager,
        llm_provider: str = "Local Extractive Synthesizer (No API Key Required)",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.similarity_threshold = similarity_threshold

    def answer_question(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        filter_filenames: Optional[List[str]] = None,
        strict_threshold: Optional[float] = None
    ) -> RAGResponse:
        """
        Processes a question through the full RAG pipeline:
        1. Query embedding & Vector Search
        2. Relevance validation (insufficient evidence check)
        3. Grounded LLM answering with source citations
        4. Structured citation mapping
        """
        threshold = strict_threshold if strict_threshold is not None else self.similarity_threshold
        question = question.strip()

        # Step 1: Embed query and search FAISS
        query_vec = self.embedding_manager.embed_query(question)
        retrieved = self.vector_store.search(
            query_vector=query_vec,
            top_k=top_k,
            filter_filenames=filter_filenames,
            min_score=None
        )

        # Step 2: Insufficient Evidence Check
        if not retrieved or (retrieved[0][1] < threshold):
            max_score = retrieved[0][1] if retrieved else 0.0
            explanation = (
                f"{UNANSWERABLE_PHRASE}\n\n"
                f"**Retrieval Diagnostic**:\n"
                f"- Highest similarity score observed: `{max_score:.1%}` (Required threshold: `{threshold:.1%}`).\n"
                f"- No sufficiently relevant passages were found in the uploaded documents to answer your inquiry.\n\n"
                f"**Recommendation**: Please check if the relevant college handbook or syllabus is uploaded, "
                f"or consult your department academic advisor."
            )
            return RAGResponse(
                question=question,
                answer=explanation,
                found_in_docs=False,
                retrieved_chunks=retrieved,
                citations=[],
                model_used=self.llm_provider
            )

        # Step 3: Format Context Passages for Grounding
        context_blocks = []
        citations_list: List[Citation] = []

        for idx, (chunk, score) in enumerate(retrieved, start=1):
            block = (
                f"--- [Passage {idx}] Source: {chunk.filename} | {chunk.section_label} | "
                f"Relevance: {score:.1%} ---\n"
                f"{chunk.text}\n"
            )
            context_blocks.append(block)

            # Build Citation object
            snippet = chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
            citations_list.append(Citation(
                tag=chunk.citation_tag,
                filename=chunk.filename,
                section_label=chunk.section_label,
                similarity_score=score,
                snippet=snippet
            ))

        formatted_context = "\n".join(context_blocks)

        # Step 4: Generate Grounded Answer via selected LLM
        prompt = (
            f"Context Passages:\n{formatted_context}\n\n"
            f"User Question: {question}\n\n"
            f"Provide a factual, well-organized answer strictly based on the Context above. "
            f"Cite the specific document and page/slide/sheet in square brackets for every fact. "
            f"If the answer is not supported by the context, respond that the answer was not found."
        )

        answer_text, model_tag = self._generate_response(prompt, context_chunks=[c for c, _ in retrieved])

        # Detect if model itself concluded unanswerable
        lower_ans = answer_text.lower()
        is_unanswerable = (
            "not found in the provided" in lower_ans or
            "not contain sufficient information" in lower_ans or
            "could not find sufficient" in lower_ans or
            "no information provided" in lower_ans or
            "is not mentioned in" in lower_ans
        )

        return RAGResponse(
            question=question,
            answer=answer_text,
            found_in_docs=not is_unanswerable,
            retrieved_chunks=retrieved,
            citations=citations_list if not is_unanswerable else [],
            model_used=model_tag
        )

    def _generate_response(self, full_prompt: str, context_chunks: List[DocumentChunk]) -> Tuple[str, str]:
        """Dispatches to chosen LLM provider or local fallback."""
        if "Gemini" in self.llm_provider:
            return self._call_gemini(full_prompt)
        elif "OpenAI" in self.llm_provider:
            return self._call_openai(full_prompt)
        elif "Groq" in self.llm_provider:
            return self._call_groq(full_prompt)
        else:
            return self._call_local_synthesizer(full_prompt, context_chunks)

    def _call_gemini(self, prompt: str) -> Tuple[str, str]:
        import google.generativeai as genai
        if not self.api_key:
            raise ValueError("Google Gemini API Key is required. Please set it in Settings.")
        genai.configure(api_key=self.api_key)
        model_name = self.model_name or "gemini-1.5-flash"
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=RAG_SYSTEM_PROMPT
        )
        response = model.generate_content(prompt)
        return response.text, f"Google Gemini ({model_name})"

    def _call_openai(self, prompt: str) -> Tuple[str, str]:
        from openai import OpenAI
        if not self.api_key:
            raise ValueError("OpenAI API Key is required. Please set it in Settings.")
        client = OpenAI(api_key=self.api_key)
        model_name = self.model_name or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content, f"OpenAI ({model_name})"

    def _call_groq(self, prompt: str) -> Tuple[str, str]:
        from openai import OpenAI
        if not self.api_key:
            raise ValueError("Groq API Key is required. Please set it in Settings.")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.api_key)
        model_name = self.model_name or "llama-3.3-70b-versatile"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content, f"Groq ({model_name})"

    def _call_local_synthesizer(self, prompt: str, context_chunks: List[DocumentChunk]) -> Tuple[str, str]:
        """
        Intelligent offline extractive synthesizer.
        Ranks candidate sentences by semantic keyword overlap with the user question,
        detects absence of evidence, and constructs a grounded answer with citations.
        """
        if not context_chunks:
            return UNANSWERABLE_PHRASE, "Local Extractive Synthesizer (Offline)"

        # Extract user query
        q_match = re.search(r"User Question:\s*(.+?)(?:\n\n|$)", prompt)
        user_q = q_match.group(1).strip() if q_match else ""

        stopwords = {
            "what", "is", "are", "the", "for", "and", "in", "to", "of", "a", "an", "on", "by",
            "at", "with", "from", "as", "how", "much", "many", "when", "where", "which", "who",
            "can", "do", "does", "student", "students", "college", "university", "policy"
        }
        q_keywords = [w.lower() for w in re.findall(r"\w+", user_q) if len(w) > 2 and w.lower() not in stopwords]

        # Score and rank sentences across all retrieved chunks
        scored_statements = []
        for chunk in context_chunks:
            lines = [l.strip() for l in chunk.text.split("\n") if l.strip()]
            for line in lines:
                if len(line) < 15 or line.startswith("Headers:"):
                    continue
                line_lower = line.lower()
                kw_matches = sum(1 for kw in q_keywords if kw in line_lower)
                scored_statements.append((kw_matches, line, chunk.citation_tag, chunk.filename))

        # Sort by keyword match descending
        scored_statements.sort(key=lambda x: x[0], reverse=True)

        # If question had keywords but top candidate line did not meet required keyword density, reject as unanswerable
        min_kw_required = 2 if len(q_keywords) >= 3 else 1
        if q_keywords and (not scored_statements or scored_statements[0][0] < min_kw_required):
            return UNANSWERABLE_PHRASE, "Local Extractive Synthesizer (Offline)"

        # Group top matching statements by document
        grouped_by_doc: Dict[str, List[Tuple[str, str]]] = {}
        top_candidates = scored_statements[:5] if scored_statements[0][0] > 0 else scored_statements[:3]

        for _, line, tag, fn in top_candidates:
            if fn not in grouped_by_doc:
                grouped_by_doc[fn] = []
            grouped_by_doc[fn].append((line, tag))

        answer_parts = ["Based on the approved institutional documents:\n"]
        for fn, stmts in grouped_by_doc.items():
            answer_parts.append(f"**From {fn}:**")
            for stmt, tag in stmts:
                clean_stmt = re.sub(r"^[-*•]\s*", "", stmt)
                answer_parts.append(f"• {clean_stmt} {tag}")
            answer_parts.append("")

        answer_parts.append(
            "\n*Note: Synthesized using local extractive verification engine grounded in retrieved document chunks.*"
        )
        return "\n".join(answer_parts), "Local Extractive Synthesizer (Offline)"

