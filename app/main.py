from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.chat_request_utils import resolve_effective_query
from app.orchestrator import run_tradepilot_pipeline
from app.response import apply_compliance_to_state, format_pipeline_answer
from app.skills.registry import REAL_SKILLS


app = FastAPI(title="TradePilot AI")
FRONTEND_INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class ChatRequest(BaseModel):
    query: Optional[str] = None
    ticker: Optional[str] = None
    original_query: Optional[str] = None
    clarification_type: Optional[str] = None
    clarification_value: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_INDEX)


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        effective_query = resolve_effective_query(
            query=req.query,
            original_query=req.original_query,
            clarification_type=req.clarification_type,
            clarification_value=req.clarification_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state = run_tradepilot_pipeline(
        query=effective_query,
        ticker=req.ticker,
        skills=REAL_SKILLS,
    )
    state = apply_compliance_to_state(state)

    return {
        "query": effective_query,
        "ticker": req.ticker,
        "answer": format_pipeline_answer(state),
        "charts": state.get("evidence", {}).get("charts", []),
        "state": state,
    }
