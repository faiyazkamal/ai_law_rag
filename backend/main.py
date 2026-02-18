from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.rag import RAGStore

app = FastAPI(title="AI Law RAG (Backend)", version="0.1")

store: RAGStore | None = None


@app.on_event("startup")
async def startup():
    global store
    store = RAGStore()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/search_sections")
async def search_sections(q: str, k: int = 6):
    if not q or len(q.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short (min 3 chars).")
    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be 1..20")

    assert store is not None
    return store.retrieve(q.strip(), k=k)
