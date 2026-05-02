from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.orchestrator import run_tradepilot_pipeline
from app.skills.registry import REAL_SKILLS


app = FastAPI(title="TradePilot AI")


class ChatRequest(BaseModel):
    query: str
    ticker: Optional[str] = None


def format_pipeline_answer(state: dict) -> str:
    guardrails = state.get("guardrails", {})
    if guardrails.get("out_of_scope"):
        return guardrails.get("message") or "This request is outside the current system scope."

    if state.get("needs_human"):
        return state.get("clarification_question") or "More information is needed."

    decision = state.get("decision") or {}
    if not decision:
        return "The pipeline did not produce a final decision."

    scope_note = guardrails.get("scope_note")

    if decision.get("type") == "comparison":
        lines = [
            "Comparison Result:",
            decision.get("comparison_summary", "No comparison summary available."),
            "",
        ]

        per_ticker = decision.get("per_ticker", {})
        for ticker, result in per_ticker.items():
            lines.append(
                f"{ticker}: {result.get('recommendation', 'N/A')} "
                f"(Confidence: {result.get('confidence', 'N/A')})"
            )

        disclaimer = decision.get("disclaimer")
        if disclaimer:
            lines.extend(["", disclaimer])

        if scope_note:
            lines = [scope_note, ""] + lines

        return "\n".join(lines)

    lines = [
        f"Recommendation: {decision.get('recommendation', 'N/A')}",
        f"Confidence: {decision.get('confidence', 'N/A')}",
        f"Risk Level: {decision.get('risk_level', 'N/A')}",
        "",
        "Reason:",
    ]

    for item in decision.get("reasoning", []):
        lines.append(f"- {item}")

    key_driver = decision.get("drivers", {}).get("key_driver")
    if key_driver:
        lines.extend(["", f"Key Driver: {key_driver}"])

    disclaimer = decision.get("disclaimer")
    if disclaimer:
        lines.extend(["", disclaimer])

    if scope_note:
        lines = [scope_note, ""] + lines

    return "\n".join(lines)


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

    return {
        "query": req.query,
        "ticker": req.ticker,
        "answer": format_pipeline_answer(state),
        "charts": state.get("evidence", {}).get("charts", []),
        "state": state,
    }
