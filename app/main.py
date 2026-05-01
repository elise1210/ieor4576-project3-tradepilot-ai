from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.orchestrator import run_tradepilot_pipeline
from app.skills.chart import run_chart_skill


app = FastAPI(title="TradePilot AI")


class ChatRequest(BaseModel):
    query: str
    ticker: Optional[str] = None


def demo_news_skill(ticker: str, query: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} received mostly positive recent coverage.",
        "items": [
            {"headline": f"{ticker} positive headline 1"},
            {"headline": f"{ticker} positive headline 2"},
        ],
        "article_count": 2,
        "query_echo": query,
    }


def demo_market_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "current_price": 220.0,
        "start_price_7d": 210.0,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
        "ma20": 215.0,
        "history": [
            {"date": "2026-04-23", "close": 210.00},
            {"date": "2026-04-24", "close": 214.20},
            {"date": "2026-04-27", "close": 212.80},
            {"date": "2026-04-28", "close": 216.40},
            {"date": "2026-04-29", "close": 215.30},
            {"date": "2026-04-30", "close": 218.70},
            {"date": "2026-05-01", "close": 220.00},
        ],
    }


def demo_fundamentals_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} has a stable large-cap business profile.",
        "market_cap_bucket": "large_cap",
    }


def demo_sentiment_skill(news_result: dict) -> dict:
    return {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
        "source_ticker": news_result.get("ticker"),
    }


DEMO_SKILLS = {
    "news": demo_news_skill,
    "market": demo_market_skill,
    "fundamentals": demo_fundamentals_skill,
    "sentiment": demo_sentiment_skill,
    "chart": run_chart_skill,
}


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
        skills=DEMO_SKILLS,
    )

    return {
        "query": req.query,
        "ticker": req.ticker,
        "answer": format_pipeline_answer(state),
        "charts": state.get("evidence", {}).get("charts", []),
        "state": state,
    }
