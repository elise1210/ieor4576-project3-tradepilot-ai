from __future__ import annotations

from collections import Counter
from typing import Iterable
from unittest.mock import patch

from app.orchestrator import run_tradepilot_pipeline

from tests.evals.benchmark_cases import BENCHMARK_CASES, FAKE_SKILLS


DETERMINISTIC_EVAL_ENV = {
    "USE_LANGGRAPH": "false",
    "USE_LLM_PLANNER": "false",
    "USE_LLM_RESEARCH": "false",
    "USE_LLM_CRITIC": "false",
    "USE_LLM_DECISION": "false",
}


def _safe_pct(numerator: int, denominator: int, empty_default: float) -> float:
    if denominator <= 0:
        return float(empty_default)
    return round((numerator / denominator) * 100.0, 2)


def called_tools_from_state(state: dict) -> list[str]:
    steps = state.get("metadata", {}).get("executed_research_steps", []) or []
    ordered = []
    seen = set()

    for step in steps:
        if not isinstance(step, dict):
            continue
        skill = step.get("skill")
        if not isinstance(skill, str):
            continue
        if skill in seen:
            continue
        seen.add(skill)
        ordered.append(skill)

    return ordered


def _has_payload(payload: object) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        return bool(payload) and not payload.get("error")
    if isinstance(payload, list):
        return bool(payload)
    return True


def _has_chart_for_ticker(state: dict, ticker: str) -> bool:
    charts = state.get("evidence", {}).get("charts", []) or []
    for payload in charts:
        if not isinstance(payload, dict):
            continue
        if payload.get("ticker") != ticker:
            continue
        if _has_payload(payload):
            return True
    return False


def evidence_present_for_case(state: dict, evidence_type: str, tickers: Iterable[str]) -> bool:
    if evidence_type == "chart":
        return all(_has_chart_for_ticker(state, ticker) for ticker in tickers)

    bucket = state.get("evidence", {}).get(evidence_type, {}) or {}
    return all(_has_payload(bucket.get(ticker)) for ticker in tickers)


def score_case(state: dict, case: dict) -> dict:
    expected_tickers = list(case.get("expected_tickers", []))
    required_tools = list(case.get("required_tools", []))
    forbidden_tools = list(case.get("forbidden_tools", []))
    required_evidence = list(case.get("required_evidence", []))

    called_tools = called_tools_from_state(state)
    called_tool_set = set(called_tools)

    required_tool_hits = len([tool for tool in required_tools if tool in called_tool_set])
    wrong_tools = [tool for tool in called_tools if tool in set(forbidden_tools)]
    evidence_hits = len(
        [
            evidence_type
            for evidence_type in required_evidence
            if evidence_present_for_case(state, evidence_type, expected_tickers)
        ]
    )

    if required_tools:
        right_tool_call_pct = _safe_pct(required_tool_hits, len(required_tools), empty_default=100.0)
    else:
        right_tool_call_pct = 100.0 if not called_tools else 0.0

    wrong_tool_call_pct = _safe_pct(len(wrong_tools), len(called_tools), empty_default=0.0)
    evidence_coverage_pct = _safe_pct(evidence_hits, len(required_evidence), empty_default=100.0)

    missing_required_tools = [tool for tool in required_tools if tool not in called_tool_set]
    missing_evidence = [
        evidence_type
        for evidence_type in required_evidence
        if not evidence_present_for_case(state, evidence_type, expected_tickers)
    ]

    intent_correct = state.get("intent") == case.get("expected_intent")
    tickers_correct = list(state.get("tickers", [])) == expected_tickers
    stop_reason_correct = (
        state.get("metadata", {}).get("stopped_reason") == case.get("expected_stop_reason")
    )

    case_pass = (
        intent_correct
        and tickers_correct
        and stop_reason_correct
        and required_tool_hits == len(required_tools)
        and not wrong_tools
        and evidence_hits == len(required_evidence)
    )

    return {
        "id": case["id"],
        "query": case["query"],
        "intent_correct": intent_correct,
        "tickers_correct": tickers_correct,
        "stop_reason_correct": stop_reason_correct,
        "called_tools": called_tools,
        "missing_required_tools": missing_required_tools,
        "wrong_tools": wrong_tools,
        "required_tool_hits": required_tool_hits,
        "required_tool_total": len(required_tools),
        "right_tool_call_pct": right_tool_call_pct,
        "wrong_tool_hits": len(wrong_tools),
        "called_tool_total": len(called_tools),
        "wrong_tool_call_pct": wrong_tool_call_pct,
        "evidence_hits": evidence_hits,
        "evidence_total": len(required_evidence),
        "evidence_coverage_pct": evidence_coverage_pct,
        "missing_evidence": missing_evidence,
        "case_pass": case_pass,
    }


def aggregate_case_scores(scorecards: list[dict]) -> dict:
    totals = Counter()
    for card in scorecards:
        totals["case_count"] += 1
        totals["intent_correct"] += int(card["intent_correct"])
        totals["tickers_correct"] += int(card["tickers_correct"])
        totals["stop_reason_correct"] += int(card["stop_reason_correct"])
        totals["required_tool_hits"] += card["required_tool_hits"]
        totals["required_tool_total"] += card["required_tool_total"]
        totals["wrong_tool_hits"] += card["wrong_tool_hits"]
        totals["called_tool_total"] += card["called_tool_total"]
        totals["evidence_hits"] += card["evidence_hits"]
        totals["evidence_total"] += card["evidence_total"]
        totals["case_pass"] += int(card["case_pass"])

    case_count = totals["case_count"]
    return {
        "case_count": case_count,
        "intent_accuracy_pct": _safe_pct(totals["intent_correct"], case_count, empty_default=0.0),
        "ticker_accuracy_pct": _safe_pct(totals["tickers_correct"], case_count, empty_default=0.0),
        "stop_reason_accuracy_pct": _safe_pct(
            totals["stop_reason_correct"],
            case_count,
            empty_default=0.0,
        ),
        "right_tool_call_pct": _safe_pct(
            totals["required_tool_hits"],
            totals["required_tool_total"],
            empty_default=100.0,
        ),
        "wrong_tool_call_pct": _safe_pct(
            totals["wrong_tool_hits"],
            totals["called_tool_total"],
            empty_default=0.0,
        ),
        "evidence_coverage_pct": _safe_pct(
            totals["evidence_hits"],
            totals["evidence_total"],
            empty_default=100.0,
        ),
        "end_to_end_success_pct": _safe_pct(totals["case_pass"], case_count, empty_default=0.0),
        "totals": dict(totals),
    }


def run_benchmark_suite(cases: list[dict] | None = None, skills: dict | None = None) -> dict:
    benchmark_cases = list(cases or BENCHMARK_CASES)
    benchmark_skills = skills or FAKE_SKILLS
    scorecards = []

    with patch.dict("os.environ", DETERMINISTIC_EVAL_ENV, clear=False):
        for case in benchmark_cases:
            state = run_tradepilot_pipeline(query=case["query"], skills=benchmark_skills)
            scorecards.append(score_case(state, case))

    return {
        "cases": scorecards,
        "summary": aggregate_case_scores(scorecards),
    }


__all__ = [
    "DETERMINISTIC_EVAL_ENV",
    "aggregate_case_scores",
    "called_tools_from_state",
    "evidence_present_for_case",
    "run_benchmark_suite",
    "score_case",
]
