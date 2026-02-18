# backend/llm_groq.py
import os
from groq import Groq


def get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set. Put it in .env (repo root).")
    return Groq(api_key=key)


def call_groq_json(prompt: str) -> str:
    """
    Returns raw text; we enforce JSON-only via system + (when available) response_format.
    """
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = get_groq_client()

    messages = [
        {"role": "system", "content": "Return ONLY valid JSON. No markdown. No extra text."},
        {"role": "user", "content": prompt},
    ]

    # Try strict JSON mode first (if supported by model/SDK)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content
