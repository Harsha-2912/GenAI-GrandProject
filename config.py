import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "faiss_index"
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"

# Ensure runtime directories exist
for path in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, SAMPLE_DOCS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Document Processing Settings
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".xlsx", ".xls", ".csv", ".pptx", ".docx"}
MAX_FILE_SIZE_MB = 25
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100

# Retrieval & Vector Store Settings
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.32  # Below this, evidence is considered insufficient
DEFAULT_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Available Providers & Models
EMBEDDING_PROVIDERS = [
    "Local (SentenceTransformers: all-MiniLM-L6-v2)",
    "Google Gemini (models/text-embedding-004)",
    "OpenAI (text-embedding-3-small)",
    "Offline / TF-IDF Vectorizer"
]

LLM_PROVIDERS = [
    "Google Gemini",
    "OpenAI",
    "Groq",
    "Local Extractive Synthesizer (No API Key Required)"
]

GEMINI_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro"
]

OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-3.5-turbo"
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

# Strict Institutional Grounding System Prompt
RAG_SYSTEM_PROMPT = """You are DocuSage AI, an institutional Document Intelligence assistant for students and staff.
Your role is to provide accurate, reliable answers strictly grounded in the approved institutional documents provided in the context.

CRITICAL INSTRUCTIONS:
1. STRICT GROUNDING: Use ONLY the factual information contained in the provided Context passages. Do NOT extrapolate, speculate, or introduce outside knowledge.
2. CITATIONS: Whenever stating a fact or guideline, cite the source document and page/slide/sheet in square brackets, for example: [Academic_Handbook.pdf, Page 4] or [Campus_Guide.pptx, Slide 3] or [Grading_Matrix.xlsx, Sheet: Grades, Row 5].
3. UNANSWERABLE QUESTIONS / INSUFFICIENT EVIDENCE: If the provided Context does not contain sufficient facts to completely and accurately answer the user's question, you MUST clearly state:
   "I could not find sufficient information in the provided approved documents to answer this question."
   Explain briefly what specific information was missing, and advise the user to consult the relevant college department or upload the appropriate handbook.
4. TONE & FORMAT: Maintain a professional, clear, and student/staff-friendly tone. Use bullet points or numbered lists where appropriate for clarity.
"""

UNANSWERABLE_PHRASE = "I could not find sufficient information in the provided approved documents to answer this question."

# Institutional Compliance & Confidentiality Notice
INSTITUTIONAL_DISCLAIMER = (
    "Security & Policy Warning: Do not upload confidential institutional documents, "
    "personally identifiable student records, or sensitive administrative files without proper institutional authorization. "
    "All document processing and vector indices are managed within your designated session."
)
