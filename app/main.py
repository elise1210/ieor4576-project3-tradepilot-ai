from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.orchestrator import run_tradepilot_pipeline
from app.response import apply_compliance_to_state, format_pipeline_answer
from app.skills.registry import REAL_SKILLS


app = FastAPI(title="TradePilot AI")


class ChatRequest(BaseModel):
    query: str
    ticker: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    state = run_tradepilot_pipeline(
        query=req.query,
        ticker=req.ticker,
        skills=REAL_SKILLS,
    )
    state = apply_compliance_to_state(state)

    return {
        "query": req.query,
        "ticker": req.ticker,
        "answer": format_pipeline_answer(state),
        "charts": state.get("evidence", {}).get("charts", []),
        "state": state,
    }
