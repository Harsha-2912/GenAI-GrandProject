import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
import numpy as np
import faiss

from src.chunker import DocumentChunk

class FaissVectorStore:
    """Production FAISS vector store with cosine search, metadata preservation, document filtering, and deletion."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_registry: Dict[str, Dict[str, Any]] = {}
        self._init_empty_index(dimension)

    def _init_empty_index(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP with normalized vectors computes exact cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.embeddings = None

    def add_chunks(self, new_chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Adds new document chunks and corresponding embeddings into the FAISS index."""
        if not new_chunks:
            return

        embeddings = np.array(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(new_chunks):
            raise ValueError(f"Mismatch: {len(new_chunks)} chunks vs {embeddings.shape[0]} embeddings.")

        emb_dim = embeddings.shape[1]
        if self.index is None or self.index.ntotal == 0:
            self._init_empty_index(emb_dim)
            self.embeddings = embeddings
        elif self.dimension != emb_dim:
            self._init_empty_index(emb_dim)
            self.chunks.clear()
            self.embeddings = embeddings
        else:
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])

        # Add to FAISS index
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

        # Update registry
        for chunk in new_chunks:
            fn = chunk.filename
            if fn not in self.document_registry:
                self.document_registry[fn] = {
                    "filename": fn,
                    "file_type": chunk.file_type,
                    "chunk_count": 0,
                    "char_count": 0,
                    "sections_covered": set(),
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            self.document_registry[fn]["chunk_count"] += 1
            self.document_registry[fn]["char_count"] += chunk.char_count
            self.document_registry[fn]["sections_covered"].add(chunk.section_label)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        filter_filenames: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top_k chunks matching query_vector.
        Supports document filtering (filter_filenames) and score thresholding.
        Returns List of (DocumentChunk, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        query_vector = np.array(query_vector, dtype=np.float32)
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        # Fetch more candidates if filtering is active
        fetch_k = min(self.index.ntotal, max(top_k * 4, 25))
        scores, indices = self.index.search(query_vector, fetch_k)

        results: List[Tuple[DocumentChunk, float]] = []
        filter_set = set(filter_filenames) if filter_filenames else None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]

            # Apply document filtering
            if filter_set and chunk.filename not in filter_set:
                continue

            # Apply minimum similarity score threshold
            sim_score = float(score)
            if min_score is not None and sim_score < min_score:
                continue

            results.append((chunk, sim_score))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, filename: str, embedding_manager=None) -> bool:
        """
        Removes all chunks of a specific document and rebuilds the FAISS index cleanly.
        """
        if filename not in self.document_registry and not any(c.filename == filename for c in self.chunks):
            return False

        # Identify indices to keep
        keep_indices = [i for i, c in enumerate(self.chunks) if c.filename != filename]
        remaining_chunks = [self.chunks[i] for i in keep_indices]
        old_embeddings = self.embeddings

        # Remove from registry
        if filename in self.document_registry:
            del self.document_registry[filename]

        # Reset index
        self._init_empty_index(self.dimension)
        self.chunks.clear()

        if remaining_chunks:
            if old_embeddings is not None and len(keep_indices) > 0 and old_embeddings.shape[0] >= len(self.chunks):
                rem_emb = old_embeddings[keep_indices]
                self.add_chunks(remaining_chunks, rem_emb)
            elif embedding_manager:
                texts = [c.text for c in remaining_chunks]
                new_embeddings = embedding_manager.embed_documents(texts)
                self.add_chunks(remaining_chunks, new_embeddings)
            else:
                self.chunks = remaining_chunks
                for chunk in remaining_chunks:
                    fn = chunk.filename
                    if fn not in self.document_registry:
                        self.document_registry[fn] = {
                            "filename": fn,
                            "file_type": chunk.file_type,
                            "chunk_count": 0,
                            "char_count": 0,
                            "sections_covered": set(),
                            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                    self.document_registry[fn]["chunk_count"] += 1
                    self.document_registry[fn]["char_count"] += chunk.char_count
                    self.document_registry[fn]["sections_covered"].add(chunk.section_label)

        return True

    def clear(self):
        """Clears all vectors, chunks, and registries."""
        self._init_empty_index(self.dimension)
        self.chunks.clear()
        self.document_registry.clear()

    def get_indexed_files_summary(self) -> List[Dict[str, Any]]:
        """Returns structured summaries of all indexed documents for UI display."""
        summary = []
        for fn, reg in self.document_registry.items():
            sections = reg.get("sections_covered", set())
            summary.append({
                "Document Name": fn,
                "Format": reg.get("file_type", "").upper(),
                "Chunks": reg.get("chunk_count", 0),
                "Sections / Pages": len(sections),
                "Characters": reg.get("char_count", 0),
                "Indexed At": reg.get("indexed_at", "N/A")
            })
        return summary

    def save_to_disk(self, folder_path: Union[str, Path]):
        """Persists FAISS index, chunk metadata, and registry to disk."""
        folder = Path(folder_path)
        folder.mkdir(parents=True, exist_ok=True)

        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, str(folder / "index.faiss"))

        # Save chunks
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(folder / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # Save registry (convert sets to lists)
        serializable_reg = {}
        for fn, info in self.document_registry.items():
            reg_copy = dict(info)
            if isinstance(reg_copy.get("sections_covered"), set):
                reg_copy["sections_covered"] = list(reg_copy["sections_covered"])
            serializable_reg[fn] = reg_copy

        with open(folder / "registry.json", "w", encoding="utf-8") as f:
            json.dump(serializable_reg, f, ensure_ascii=False, indent=2)

    def load_from_disk(self, folder_path: Union[str, Path]) -> bool:
        """Loads FAISS index, chunk metadata, and registry from disk."""
        folder = Path(folder_path)
        index_file = folder / "index.faiss"
        chunks_file = folder / "chunks.json"
        reg_file = folder / "registry.json"

        if not (index_file.exists() and chunks_file.exists() and reg_file.exists()):
            return False

        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d

            with open(chunks_file, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
                self.chunks = [DocumentChunk.from_dict(d) for d in raw_chunks]

            with open(reg_file, "r", encoding="utf-8") as f:
                raw_reg = json.load(f)
                self.document_registry = {}
                for fn, info in raw_reg.items():
                    info["sections_covered"] = set(info.get("sections_covered", []))
                    self.document_registry[fn] = info

            return True
        except Exception as e:
            print(f"[FaissVectorStore] Error loading from disk: {e}")
            return False
