import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"

    chunks_path = data / "chunks.jsonl"
    index_path = data / "faiss.index"
    docstore_path = data / "docstore.json"

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)

    embeddings = []
    for t in tqdm(texts, desc="Embedding"):
        v = model.encode(t, normalize_embeddings=True)
        embeddings.append(v)

    X = np.vstack(embeddings).astype("float32")
    dim = X.shape[1]

    # cosine similarity because embeddings are normalized
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    faiss.write_index(index, str(index_path))
    docstore_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved:")
    print(" -", index_path)
    print(" -", docstore_path)
    print("Vectors:", index.ntotal, "Dim:", dim)


if __name__ == "__main__":
    main()
