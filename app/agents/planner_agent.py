import re
import os
from typing import Optional, Tuple

from dotenv import load_dotenv

from app.agents.llm_planner import run_llm_planner
from app.skills.chart import should_show_chart_for_query
from app.state import DEFAULT_MAX_ITERATIONS, clone_state


load_dotenv()


COMPANY_NAME_TO_TICKER = {
    "apple": "AAPL",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "alphabet": "GOOGL",
    "google": "GOOGL",
}

TICKER_TO_TERMS = {}
for _company_name, _ticker in COMPANY_NAME_TO_TICKER.items():
    TICKER_TO_TERMS.setdefault(_ticker, []).append(_company_name)

EXPLICIT_TICKER_STOPWORDS = {
    "I",
    "A",
    "AN",
    "THE",
    "BUY",
    "SELL",
    "HOLD",
    "AND",
    "OR",
    "VS",
}

COMPARISON_KEYWORDS = ("compare", "versus", "vs", "better than")
BUY_SELL_KEYWORDS = ("buy", "sell", "hold", "should i buy", "should i sell")
EXPLANATION_KEYWORDS = ("why", "what happened", "explain", "summarize")
SUMMARY_RESEARCH_KEYWORDS = (
    "summary",
    "summarize",
    "news",
    "headline",
    "headlines",
    "update",
    "updates",
    "developments",
)
FINANCE_CONTEXT_KEYWORDS = (
    "stock",
    "stocks",
    "share",
    "shares",
    "market",
    "price",
    "prices",
    "ticker",
    "company",
    "earnings",
    "invest",
    "investment",
    "portfolio",
)

SHORT_TERM_MARKERS = ("today", "this week", "this month", "short term", "near term")
LONG_TERM_MARKERS = ("long term", "next year", "5 years", "multi year", "years")
FUTURE_SCOPE_LIMITS = {
    "tomorrow": "day",
    "this week": "week",
    "next week": "week",
    "this month": "month",
    "next month": "month",
    "next year": "year",
    "long term": "long-term horizon",
    "5 years": "multi-year horizon",
    "multi year": "multi-year horizon",
}

EVIDENCE_BY_INTENT = {
    "buy_sell_decision": ["news", "market", "fundamentals", "sentiment"],
    "comparison": ["news", "market", "fundamentals", "sentiment"],
    "explanation": ["news", "market", "fundamentals"],
    "general_research": ["news", "market", "fundamentals"],
}


def classify_intent(query: str) -> str:
    text = (query or "").strip().lower()

    if any(keyword in text for keyword in COMPARISON_KEYWORDS):
        return "comparison"
    if any(keyword in text for keyword in BUY_SELL_KEYWORDS):
        return "buy_sell_decision"
    if any(keyword in text for keyword in EXPLANATION_KEYWORDS):
        return "explanation"
    return "general_research"


def refine_intent(query: str, intent: str) -> str:
    text = (query or "").strip().lower()

    if intent in {"buy_sell_decision", "comparison"}:
        return intent

    if any(keyword in text for keyword in EXPLANATION_KEYWORDS):
        return "explanation"

    if any(keyword in text for keyword in SUMMARY_RESEARCH_KEYWORDS):
        return "explanation"

    return intent


def infer_time_horizon(query: str) -> str:
    text = (query or "").strip().lower()

    if any(marker in text for marker in SHORT_TERM_MARKERS):
        return "short_term"
    if any(marker in text for marker in LONG_TERM_MARKERS):
        return "long_term"
    return "unknown"


def extract_explicit_tickers(query: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z]{1,5}\b", query or "")
    tickers = []

    for candidate in candidates:
        if candidate in EXPLICIT_TICKER_STOPWORDS:
            continue
        if candidate not in tickers:
            tickers.append(candidate)

    return tickers


def infer_tickers_from_company_names(query: str) -> list[str]:
    text = (query or "").lower()
    tickers = []

    # Match longer names first so "advanced micro devices" wins before "amd".
    for company_name, ticker in sorted(
        COMPANY_NAME_TO_TICKER.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if company_name in text and ticker not in tickers:
            tickers.append(ticker)

    return tickers


def order_tickers_by_query_appearance(query: str, tickers: list[str]) -> list[str]:
    text = (query or "").lower()
    positions = {}

    for ticker in tickers:
        candidate_positions = []

        explicit_index = text.find(ticker.lower())
        if explicit_index != -1:
            candidate_positions.append(explicit_index)

        for term in TICKER_TO_TERMS.get(ticker, []):
            idx = text.find(term)
            if idx != -1:
                candidate_positions.append(idx)

        positions[ticker] = min(candidate_positions) if candidate_positions else 10**9

    return sorted(tickers, key=lambda ticker: positions[ticker])


def infer_tickers(
    query: str,
    provided_ticker: Optional[str] = None,
) -> Tuple[list[str], str, Optional[str]]:
    if provided_ticker:
        return [provided_ticker.upper()], "provided", "high"

    explicit = extract_explicit_tickers(query)
    inferred = infer_tickers_from_company_names(query)

    merged = []
    for ticker in explicit + inferred:
        if ticker not in merged:
            merged.append(ticker)
    merged = order_tickers_by_query_appearance(query, merged)

    if explicit and inferred:
        return merged, "mixed_query", "high"
    if explicit:
        return merged, "explicit_query", "high"
    if inferred:
        if len(inferred) == 1:
            return merged, "company_name", "medium"
        return merged, "company_name_multi", "medium"

    return [], "unknown", None


def build_task_plan(intent: str, query: str = "") -> dict:
    required = list(EVIDENCE_BY_INTENT.get(intent, ["news", "market", "fundamentals"]))
    if should_show_chart_for_query(query) and "chart" not in required:
        required.append("chart")

    return {
        "required_evidence": required,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
    }


def is_out_of_scope(query: str, tickers: list[str], intent: str) -> bool:
    text = (query or "").strip().lower()

    if tickers:
        return False
    if intent != "general_research":
        return False
    if any(keyword in text for keyword in FINANCE_CONTEXT_KEYWORDS):
        return False

    return True


def build_out_of_scope_message() -> str:
    return (
        "I can help with informational stock analysis, company comparisons, "
        "and daily buy/hold/sell-style signals based on the latest available data."
    )


def build_scope_note(query: str, intent: str) -> Optional[str]:
    if intent != "buy_sell_decision":
        return None

    text = (query or "").strip().lower()
    for marker, label in FUTURE_SCOPE_LIMITS.items():
        if marker in text:
            if label == "day":
                return (
                    "I cannot forecast tomorrow in advance. I can provide a daily "
                    "informational buy/hold/sell signal based on the latest available data."
                )
            return (
                f"I cannot forecast the full {label}. I can provide a daily informational "
                "buy/hold/sell signal based on the latest available data."
            )

    return None


def build_clarification_question(
    intent: str,
    tickers: list[str],
    time_horizon: str,
) -> Optional[str]:
    if not tickers:
        return "Which company or ticker do you want me to analyze?"

    if intent == "buy_sell_decision" and time_horizon == "unknown":
        if len(tickers) == 1:
            return (
                f"I inferred {tickers[0]}. Do you want a short-term trading view "
                "or a longer-term investment view?"
            )
        return "Do you want a short-term trading view or a longer-term investment view?"

    return None


def _apply_planner_result(
    state: dict,
    intent: str,
    tickers: list[str],
    time_horizon: str,
    ticker_source: str,
    confidence: Optional[str],
    clarification_question: Optional[str] = None,
    planner_mode: Optional[str] = None,
    planner_reasoning_brief: Optional[str] = None,
) -> dict:
    next_state = clone_state(state)

    query = next_state.get("query", "")
    plan = build_task_plan(intent, query=query)
    clarification_question = clarification_question or build_clarification_question(
        intent,
        tickers,
        time_horizon,
    )
    out_of_scope = is_out_of_scope(query, tickers, intent)
    scope_note = build_scope_note(query, intent)

    next_state["intent"] = intent
    next_state["tickers"] = tickers
    next_state["time_horizon"] = time_horizon
    next_state["plan"] = plan
    next_state["needs_human"] = clarification_question is not None
    next_state["clarification_question"] = clarification_question
    next_state["guardrails"]["out_of_scope"] = out_of_scope
    next_state["guardrails"]["message"] = build_out_of_scope_message() if out_of_scope else None
    next_state["guardrails"]["scope_note"] = scope_note
    next_state["metadata"]["ticker_source"] = ticker_source
    next_state["metadata"]["ticker_inference_confidence"] = confidence
    next_state["metadata"]["planner_mode"] = planner_mode
    next_state["metadata"]["planner_reasoning_brief"] = planner_reasoning_brief

    return next_state


def _run_deterministic_planner(state: dict) -> dict:
    query = state.get("query", "")
    provided_ticker = state.get("user_inputs", {}).get("provided_ticker")

    intent = classify_intent(query)
    intent = refine_intent(query, intent)
    tickers, ticker_source, confidence = infer_tickers(query, provided_ticker)
    time_horizon = infer_time_horizon(query)

    return _apply_planner_result(
        state=state,
        intent=intent,
        tickers=tickers,
        time_horizon=time_horizon,
        ticker_source=ticker_source,
        confidence=confidence,
        planner_mode="deterministic_only",
    )


def _llm_planner_enabled() -> bool:
    value = os.getenv("USE_LLM_PLANNER")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return bool(os.getenv("OPENAI_API_KEY"))


def run_planner_agent(state: dict) -> dict:
    if not _llm_planner_enabled():
        return _run_deterministic_planner(state)

    query = state.get("query", "")
    provided_ticker = state.get("user_inputs", {}).get("provided_ticker")
    llm_result = run_llm_planner(
        query=query,
        provided_ticker=provided_ticker,
        company_name_to_ticker=COMPANY_NAME_TO_TICKER,
    )

    if llm_result is None:
        fallback_state = _run_deterministic_planner(state)
        fallback_state["metadata"]["planner_mode"] = "deterministic_fallback"
        return fallback_state

    clarification_question = llm_result.get("clarification_question")
    if llm_result.get("needs_human") and clarification_question is None:
        clarification_question = build_clarification_question(
            llm_result["intent"],
            llm_result["tickers"],
            llm_result["time_horizon"],
        )

    return _apply_planner_result(
        state=state,
        intent=refine_intent(query, llm_result["intent"]),
        tickers=llm_result["tickers"],
        time_horizon=llm_result["time_horizon"],
        ticker_source=llm_result["ticker_source"],
        confidence=llm_result["ticker_inference_confidence"],
        clarification_question=clarification_question,
        planner_mode="llm",
        planner_reasoning_brief=llm_result.get("reasoning_brief"),
    )
