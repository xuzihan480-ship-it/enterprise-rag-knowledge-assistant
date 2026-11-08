"""New trust-aware FastAPI variant; does NOT replace app_adaptive.py.

Start: python -m uvicorn app_verified:app --host 127.0.0.1 --port 8001
Needs vLLM-Metal on port 8000 and prior evidence_gate_config.json.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from verified_rag import VerifiedRAG

app = FastAPI(title='Verified Adaptive Knowledge Assistant', version='0.4.0')
engine = VerifiedRAG()

class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'hybrid_adaptive_evidence_verified', 'indexed_chunks': len(engine.chunks)}

@app.post('/ask')
def ask(request: QuestionRequest):
    try:
        result = engine.answer(request.question)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result['answerability_decision'] == 'error':
        raise HTTPException(status_code=503, detail='Local vLLM verification service failed: ' + str(result.get('refusal_reason')))
    return result
