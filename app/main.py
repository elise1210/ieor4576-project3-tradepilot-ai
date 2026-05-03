from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.chat_request_utils import resolve_effective_query
from app.graph.runtime import resume_tradepilot_graph_run, start_tradepilot_graph_run
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


class ChatStartRequest(BaseModel):
    query: str
    ticker: Optional[str] = None


class ChatResumeRequest(BaseModel):
    thread_id: str
    clarification_value: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def frontend():
    return FileResponse(FRONTEND_INDEX)


def _response_payload(
    *,
    query: str,
    ticker: Optional[str],
    state: dict,
    thread_id: Optional[str] = None,
    status: Optional[str] = None,
) -> dict:
    safe_state = apply_compliance_to_state(state)
    if thread_id:
        safe_state.setdefault("metadata", {})["graph_thread_id"] = thread_id
    return {
        "query": query,
        "ticker": ticker,
        "thread_id": thread_id,
        "status": status or safe_state.get("metadata", {}).get("stopped_reason"),
        "answer": format_pipeline_answer(safe_state),
        "charts": safe_state.get("evidence", {}).get("charts", []),
        "state": safe_state,
    }


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
    return _response_payload(query=effective_query, ticker=req.ticker, state=state)


@app.post("/chat/start")
def chat_start(req: ChatStartRequest):
    try:
        thread_id, state, _, status = start_tradepilot_graph_run(
            query=req.query,
            ticker=req.ticker,
            skills=REAL_SKILLS,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="LangGraph runtime is not available. Install the project requirements first.",
        ) from exc

    return _response_payload(
        query=req.query,
        ticker=req.ticker,
        state=state,
        thread_id=thread_id,
        status=status,
    )


@app.post("/chat/resume")
def chat_resume(req: ChatResumeRequest):
    if not req.thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id is required")
    if not req.clarification_value.strip():
        raise HTTPException(status_code=400, detail="clarification_value is required")

    try:
        thread_id, state, _, status = resume_tradepilot_graph_run(
            thread_id=req.thread_id.strip(),
            clarification_value=req.clarification_value,
            skills=REAL_SKILLS,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="LangGraph runtime is not available. Install the project requirements first.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not resume that analysis. Start a new run if the session was lost.",
        ) from exc

    return _response_payload(
        query=state.get("query", ""),
        ticker=(state.get("tickers") or [None])[0],
        state=state,
        thread_id=thread_id,
        status=status,
    )
