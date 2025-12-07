
from __future__ import annotations
import faiss
import json
import os
from typing import Dict, Any, List
from agent_app.core.agent.interfaces import VectorStoreInterface
from agent_app.services.storage.vector_db.embeddings import embed_text


class FAISSVectorStore(VectorStoreInterface):
    """
    Local FAISS index + JSON metadata store.
    """

    def __init__(self, index_path: str = "data/faiss.index", meta_path: str = "data/faiss_meta.json"):
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.meta = {}

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)

    async def add_document(self, text: str, metadata: Dict[str, Any] | None = None) -> str:
        vec = (await embed_text([text]))[0]
        import numpy as np
        np_vec = np.array(vec).astype("float32").reshape(1, -1)

        doc_id = metadata.get("id") if metadata else f"faiss_{hash(text)}"

        if self.index is None:
            self.index = faiss.IndexFlatL2(np_vec.shape[1])

        self.index.add(np_vec)
        self.meta[doc_id] = {"text": text, "metadata": metadata}

        self._save()
        return doc_id

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        import numpy as np

        if self.index is None:
            return []

        q_vec = (await embed_text([query]))[0]
        np_vec = np.array(q_vec).astype("float32").reshape(1, -1)

        distances, indices = self.index.search(np_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            for doc_id, meta in self.meta.items():
                results.append({
                    "id": doc_id,
                    "text": meta["text"],
                    "metadata": meta["metadata"],
                    "score": float(dist)
                })
        return results[:k]
