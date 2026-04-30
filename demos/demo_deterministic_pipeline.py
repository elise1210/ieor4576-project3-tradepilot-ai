import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.agents.critic_agent import run_critic_agent
from app.agents.decision_agent import run_decision_agent
from app.agents.planner_agent import run_planner_agent
from app.agents.research_agent import run_research_agent
from app.state import build_initial_state


def fake_news_skill(ticker: str, query: str) -> dict:
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


def fake_market_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "trend_7d": 0.03,
        "trend_label": "upward",
        "volatility": 0.018,
    }


def fake_fundamentals_skill(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "summary": f"{ticker} has a stable large-cap business profile.",
        "market_cap_bucket": "large_cap",
    }


def fake_sentiment_skill(news_result: dict) -> dict:
    return {
        "sentiment": "positive",
        "score": 0.42,
        "dispersion": 0.10,
        "source_ticker": news_result.get("ticker"),
    }


def print_section(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2))


def summarize_state(state: dict) -> dict:
    return {
        "query": state.get("query"),
        "intent": state.get("intent"),
        "tickers": state.get("tickers"),
        "time_horizon": state.get("time_horizon"),
        "plan": state.get("plan"),
        "gaps": state.get("gaps"),
        "needs_human": state.get("needs_human"),
        "clarification_question": state.get("clarification_question"),
        "critic_result": state.get("critic_result"),
        "confidence": state.get("confidence"),
        "evidence_keys": {
            "news": list(state.get("evidence", {}).get("news", {}).keys()),
            "market": list(state.get("evidence", {}).get("market", {}).keys()),
            "fundamentals": list(state.get("evidence", {}).get("fundamentals", {}).keys()),
            "sentiment": list(state.get("evidence", {}).get("sentiment", {}).keys()),
            "charts": len(state.get("evidence", {}).get("charts", [])),
        },
        "decision": state.get("decision"),
    }


def main() -> None:
    state = build_initial_state("Should I buy Apple this week?")
    print_section("Initial State", summarize_state(state))

    planner_input = summarize_state(state)
    state = run_planner_agent(state)
    print_section("Planner Input", planner_input)
    print_section("Planner Output", summarize_state(state))

    research_input_1 = summarize_state(state)
    state = run_research_agent(
        state,
        skills={
            "news": fake_news_skill,
            "market": fake_market_skill,
            "sentiment": fake_sentiment_skill,
        },
    )
    print_section("Research Pass 1 Input", research_input_1)
    print_section("Research Pass 1 Output", summarize_state(state))

    critic_input_1 = summarize_state(state)
    state = run_critic_agent(state)
    print_section("Critic Pass 1 Input", critic_input_1)
    print_section("Critic Pass 1 Output", summarize_state(state))

    research_input_2 = summarize_state(state)
    state = run_research_agent(
        state,
        skills={
            "news": fake_news_skill,
            "market": fake_market_skill,
            "fundamentals": fake_fundamentals_skill,
            "sentiment": fake_sentiment_skill,
        },
    )
    print_section("Research Pass 2 Input", research_input_2)
    print_section("Research Pass 2 Output", summarize_state(state))

    critic_input_2 = summarize_state(state)
    state = run_critic_agent(state)
    print_section("Critic Pass 2 Input", critic_input_2)
    print_section("Critic Pass 2 Output", summarize_state(state))

    decision_input = summarize_state(state)
    state = run_decision_agent(state)
    print_section("Decision Input", decision_input)
    print_section("Decision Output", summarize_state(state))


if __name__ == "__main__":
    main()
