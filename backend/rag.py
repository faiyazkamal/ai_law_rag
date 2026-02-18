import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGStore:
    """
    Loads:
      - FAISS index: data/faiss.index
      - Docstore:   data/docstore.json
    Uses sentence-transformers to embed query for retrieval.
    """

    def __init__(self):
        index_path = DATA_DIR / "faiss.index"
        docstore_path = DATA_DIR / "docstore.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")
        if not docstore_path.exists():
            raise FileNotFoundError(f"Missing docstore: {docstore_path}")

        self.index = faiss.read_index(str(index_path))
        self.docstore = json.loads(docstore_path.read_text(encoding="utf-8"))
        self.encoder = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        qv = self.encoder.encode(query, normalize_embeddings=True).astype("float32")
        qv = np.expand_dims(qv, axis=0)

        scores, idxs = self.index.search(qv, k)

        results = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            d = self.docstore[int(i)]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": d.get("chunk_id"),
                    "section_kind": d.get("section_kind"),
                    "section_id": d.get("section_id"),
                    "section_title": d.get("section_title"),
                    "text": d.get("text"),
                }
            )
        return results
