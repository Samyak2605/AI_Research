import os
import json
import pickle
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as _e:  # pragma: no cover
    faiss = None  # allow import without faiss for environments missing wheels


class LocalFaissStore:
    """Lightweight FAISS-backed vector store with on-disk persistence.

    Stores metadata alongside vectors in a parallel Python list; suitable for small
    personal research corpora. Not optimized for concurrency or massive scale.
    """

    def __init__(self, dim: int, index_path: str) -> None:
        if faiss is None:
            raise ImportError("faiss is not available. Install faiss-cpu to use LocalFaissStore.")
        self.dim = int(dim)
        self.index_path = index_path
        self.meta_path = index_path + ".meta.pkl"
        if os.path.exists(index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(index_path)
            with open(self.meta_path, "rb") as f:
                self.metadata: List[Dict] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]) -> int:
        if not embeddings:
            return 0
        arr = np.array(embeddings, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            # Attempt to infer dim on first add
            if self.index.ntotal == 0 and arr.ndim == 2:
                self.dim = int(arr.shape[1])
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                raise ValueError(f"Embedding dimension mismatch: got {arr.shape}")
        arr = self._normalize(arr)
        self.index.add(arr)
        self.metadata.extend(metadatas)
        self._save()
        return len(embeddings)

    def search(self, query_embeddings: List[List[float]], k: int = 5) -> List[List[Tuple[float, Dict]]]:
        if not query_embeddings:
            return [[]]
        q = np.array(query_embeddings, dtype="float32")
        q = self._normalize(q)
        D, I = self.index.search(q, min(max(1, k), max(1, self.index.ntotal)))
        results: List[List[Tuple[float, Dict]]] = []
        for d_row, i_row in zip(D, I):
            items: List[Tuple[float, Dict]] = []
            for dist, idx in zip(d_row.tolist(), i_row.tolist()):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                items.append((float(dist), self.metadata[idx]))
            results.append(items)
        return results


