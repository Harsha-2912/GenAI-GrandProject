import os
import numpy as np
from typing import List, Union, Optional

class EmbeddingManager:
    """Manages document and query vector generation across local models, Gemini, OpenAI, and TF-IDF."""

    _local_st_model = None

    def __init__(self, provider: str = "Local (SentenceTransformers: all-MiniLM-L6-v2)", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.tfidf_vectorizer = None
        self.tfidf_vocab_size = 512

    @classmethod
    def get_local_sentence_transformer(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._local_st_model is None:
            from sentence_transformers import SentenceTransformer
            cls._local_st_model = SentenceTransformer(model_name)
        return cls._local_st_model

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts and returns an (N, D) float32 numpy array with L2 normalization."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if "SentenceTransformers" in self.provider or "Local" in self.provider:
            return self._embed_local(texts)
        elif "Gemini" in self.provider:
            return self._embed_gemini(texts)
        elif "OpenAI" in self.provider:
            return self._embed_openai(texts)
        else:
            return self._embed_tfidf(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 normalized vector."""
        vecs = self.embed_documents([query])
        return vecs[0]

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes vectors so dot product equals cosine similarity."""
        vectors = np.array(vectors, dtype=np.float32)
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (vectors / norms).astype(np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        try:
            model = self.get_local_sentence_transformer()
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return self._normalize(embeddings)
        except Exception as e:
            # Fallback to TF-IDF if local weight download or memory fails
            print(f"[EmbeddingManager] Warning: Local model fallback due to {e}")
            return self._embed_tfidf(texts)

    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        import google.generativeai as genai
        if not self.api_key:
            raise ValueError("Google Gemini API Key is required for Gemini embeddings.")
        genai.configure(api_key=self.api_key)

        vectors = []
        for text in texts:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            vectors.append(res['embedding'])
        return self._normalize(np.array(vectors, dtype=np.float32))

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        from openai import OpenAI
        if not self.api_key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
        client = OpenAI(api_key=self.api_key)

        # Batch in chunks of 50
        batch_size = 50
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            for item in response.data:
                vectors.append(item.embedding)
        return self._normalize(np.array(vectors, dtype=np.float32))

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        """Lightweight dense TF-IDF projection for offline/zero-dependency vectorization."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        if self.tfidf_vectorizer is None or getattr(self, "_tfidf_fitted", False) is False:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=self.tfidf_vocab_size, stop_words="english")
            try:
                tfidf_mat = self.tfidf_vectorizer.fit_transform(texts).toarray()
                self._tfidf_fitted = True
            except ValueError:
                # If texts are empty or very small
                self.tfidf_vectorizer = TfidfVectorizer(max_features=self.tfidf_vocab_size)
                tfidf_mat = self.tfidf_vectorizer.fit_transform(texts).toarray()
                self._tfidf_fitted = True
        else:
            tfidf_mat = self.tfidf_vectorizer.transform(texts).toarray()

        # Ensure consistent shape (pad to 384 dimensions)
        target_dim = 384
        if tfidf_mat.shape[1] < target_dim:
            padded = np.zeros((tfidf_mat.shape[0], target_dim), dtype=np.float32)
            padded[:, :tfidf_mat.shape[1]] = tfidf_mat
            tfidf_mat = padded
        elif tfidf_mat.shape[1] > target_dim:
            tfidf_mat = tfidf_mat[:, :target_dim]

        return self._normalize(tfidf_mat)
