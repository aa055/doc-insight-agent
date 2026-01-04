import json
import pickle
import requests
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document

from src.config import settings


class OllamaEmbedder:
    """Embeddings using Ollama HTTP API directly."""

    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.EMBEDDING_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._dimension = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120
            )
            if response.status_code == 200:
                emb = response.json()["embedding"]
                embeddings.append(emb)
                if self._dimension is None:
                    self._dimension = len(emb)
            else:
                raise Exception(f"Ollama embedding error: {response.status_code}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Get embedding dimension (calls Ollama once if not known)."""
        if self._dimension is None:
            self.embed(["test"])
        return self._dimension


class VectorStoreManager:
    """Manages FAISS vector store operations with Ollama embeddings."""

    def __init__(self):
        """Initialize the vector store manager."""
        self.embedder = OllamaEmbedder()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Document] = []
        self.index_path = settings.CHROMA_PERSIST_DIR / "faiss.index"
        self.docs_path = settings.CHROMA_PERSIST_DIR / "documents.pkl"
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load existing FAISS index."""
        settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists() and self.docs_path.exists():
            # Load existing index
            print("    Loading existing FAISS index...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print(f"    Loaded {len(self.documents)} documents")
        else:
            # Create new index (will be initialized on first add)
            self.index = None
            self.documents = []

    def _save_index(self):
        """Save FAISS index and documents to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.docs_path, "wb") as f:
                pickle.dump(self.documents, f)

    def add_documents(self, documents: List[Document], batch_size: int = 10) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to embed at a time

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        all_ids = []
        total = len(documents)

        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size

            print(f"    Processing batch {batch_num}/{total_batches} ({len(batch)} docs)...")

            # Extract texts
            texts = [doc.page_content for doc in batch]

            # Generate embeddings
            print(f"      Generating embeddings...")
            embeddings = self.embedder.embed(texts)
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Normalize for cosine similarity (IndexFlatIP with normalized vectors = cosine)
            faiss.normalize_L2(embeddings_np)

            # Initialize index if needed
            if self.index is None:
                dim = embeddings_np.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                print(f"      Created FAISS index with dimension {dim}")

            # Add to index
            print(f"      Adding to FAISS...")
            start_id = len(self.documents)
            self.index.add(embeddings_np)
            self.documents.extend(batch)

            ids = [f"doc_{start_id + j}" for j in range(len(batch))]
            all_ids.extend(ids)
            print(f"      Done with batch {batch_num}")

        # Save to disk
        self._save_index()
        print(f"    Saved index with {len(self.documents)} total documents")

        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of similar Document objects
        """
        if self.index is None or len(self.documents) == 0:
            return []

        k = k or settings.RETRIEVAL_K

        # Get query embedding
        query_embedding = np.array([self.embedder.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(k * 2, len(self.documents)))

        # Filter and collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            # Apply filter if provided
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue

            results.append(doc)
            if len(results) >= k:
                break

        return results

    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get a simple retriever function."""
        k = (search_kwargs or {}).get("k", settings.RETRIEVAL_K)

        def retrieve(query: str) -> List[Document]:
            return self.similarity_search(query, k=k)

        return retrieve

    def get_all_sources(self) -> List[str]:
        """Get list of all unique document sources."""
        sources = set()
        for doc in self.documents:
            if doc.metadata and "doc_name" in doc.metadata:
                sources.add(doc.metadata["doc_name"])
        return sorted(list(sources))

    def clear_all(self):
        """Clear all documents from the store."""
        self.index = None
        self.documents = []

        # Delete files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.docs_path.exists():
            self.docs_path.unlink()

    def get_document_count(self) -> int:
        """Get the total number of documents in the store."""
        return len(self.documents)
