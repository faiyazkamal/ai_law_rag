# backend/main.py
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.rag import RAGStore, build_analysis_prompt
from backend.llm_groq import call_groq_json
from backend.db import init_db, SessionLocal, QueryLog, Feedback

# Always load .env from repo root (works no matter where uvicorn is started from)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(title="AI Law RAG (Groq Backend)", version="0.4")

store: RAGStore | None = None


class AnalyzeRequest(BaseModel):
    incident: str = Field(..., min_length=20, max_length=8000)
    k: int = Field(6, ge=3, le=12)


class FeedbackRequest(BaseModel):
    query_id: int = Field(..., ge=1)
    user_comment: str = Field(..., min_length=3, max_length=2000)
    corrected_section: str | None = Field(default=None, max_length=200)


@app.on_event("startup")
async def startup():
    global store
    init_db()
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


def parse_json_strict(s: str) -> dict:
    """
    Parse JSON robustly:
    1) direct json.loads
    2) extract first {...} region and parse
    """
    try:
        return json.loads(s)
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        return json.loads(candidate)

    raise ValueError("No JSON object found in model output")


@app.post("/analyze_incident")
async def analyze_incident(req: AnalyzeRequest):
    assert store is not None

    incident = req.incident.strip()
    retrieved = store.retrieve(incident, k=req.k)

    prompt = build_analysis_prompt(incident, retrieved)
    raw = call_groq_json(prompt)

    try:
        parsed = parse_json_strict(raw)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Groq returned non-JSON. Raw output starts with: {raw[:200]!r}",
        )

    # Attach retrieval meta for debugging
    parsed["_retrieval"] = [{"chunk_id": r["chunk_id"], "score": r["score"]} for r in retrieved]

    # Save query + response
    db = SessionLocal()
    try:
        row = QueryLog(incident=incident, response_json=json.dumps(parsed, ensure_ascii=False))
        db.add(row)
        db.commit()
        db.refresh(row)
        parsed["_query_id"] = row.id
    finally:
        db.close()

    return parsed


@app.get("/history")
async def history(limit: int = 20):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be 1..100")

    db = SessionLocal()
    try:
        rows = db.query(QueryLog).order_by(QueryLog.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "incident": r.incident[:300] + ("..." if len(r.incident) > 300 else ""),
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    finally:
        db.close()


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    db = SessionLocal()
    try:
        q = db.query(QueryLog).filter(QueryLog.id == req.query_id).first()
        if not q:
            raise HTTPException(status_code=404, detail="query_id not found")

        fb = Feedback(
            query_log_id=req.query_id,
            user_comment=req.user_comment.strip(),
            corrected_section=(req.corrected_section.strip() if req.corrected_section else None),
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        return {"ok": True, "feedback_id": fb.id}
    finally:
        db.close()
