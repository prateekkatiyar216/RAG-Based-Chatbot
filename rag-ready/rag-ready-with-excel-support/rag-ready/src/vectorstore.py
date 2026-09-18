import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def _faiss_path(self):
        return os.path.join(self.persist_dir, "faiss.index")

    @property
    def _meta_path(self):
        return os.path.join(self.persist_dir, "metadata.pkl")

    def is_built(self) -> bool:
        return os.path.exists(self._faiss_path) and os.path.exists(self._meta_path)

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building index from {len(documents)} documents...")
        pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        chunks = pipe.chunk_documents(documents)
        embeddings = pipe.embed_chunks(chunks)
        metadatas = [{"text": c.page_content, "source": c.metadata.get("source", "")} for c in chunks]
        self._add(embeddings, metadatas)
        self.save()
        print(f"[INFO] Index saved to {self.persist_dir} ({len(chunks)} chunks)")

    def _add(self, embeddings: np.ndarray, metadatas: List[Any]):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, self._faiss_path)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        self.index = faiss.read_index(self._faiss_path)
        with open(self._meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Index loaded ({self.index.ntotal} vectors)")

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        query_emb = self.model.encode([query_text]).astype("float32")
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata):
                results.append({"distance": float(dist), "metadata": self.metadata[idx]})
        return results
