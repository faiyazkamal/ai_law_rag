# backend/rag.py
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

        results: List[Dict[str, Any]] = []
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


def build_analysis_prompt(incident: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    Build a strict JSON-only prompt.
    The model MUST cite chunk_id + short quotes from the provided text.
    """
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[{r['chunk_id']}] {r['section_kind']} {r['section_id']}: {r['section_title']}\n{r['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""
You are an assistant that maps an incident to the organization's law manual.
You MUST ONLY use the provided CONTEXT. If the context is insufficient, say so.

INCIDENT:
{incident}

CONTEXT:
{context}

Return ONLY valid JSON with EXACT keys:

{{
  "matched_sections": [
    {{
      "chunk_id": "CHAPTER-...-PARA-...",
      "section_id": "e.g., I.4",
      "section_kind": "Paragraph/ChapterIntro/...",
      "section_title": "...",
      "why_applies": "short reasoning using only context",
      "exact_quotes": ["quote1", "quote2"]
    }}
  ],
  "punishment": null,
  "compensation": null,
  "confidence": 0,
  "notes": ""
}}

Rules:
- exact_quotes must be copied from CONTEXT (max 2 quotes, each <= 25 words)
- If punishment/compensation not found in CONTEXT, keep them null and explain in notes.
- confidence must be 0-100 integer.
""".strip()
