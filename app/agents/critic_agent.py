from app.state import clone_state


def _has_usable_payload(payload: object) -> bool:
    if payload is None:
        return False
    if isinstance(payload, dict):
        return bool(payload) and not payload.get("error")
    if isinstance(payload, list):
        return bool(payload)
    return True


def _required_evidence(state: dict) -> list[str]:
    return list(state.get("plan", {}).get("required_evidence", []))


def _missing_evidence_items(state: dict) -> list[str]:
    required = _required_evidence(state)
    tickers = list(state.get("tickers", []))
    evidence = state.get("evidence", {})
    missing = []

    for ticker in tickers:
        for evidence_type in required:
            if evidence_type == "chart":
                continue
            payload = evidence.get(evidence_type, {}).get(ticker, {})
            if not _has_usable_payload(payload):
                missing.append(f"{evidence_type}:{ticker}")

    return missing


def _comparison_fairness_issues(state: dict) -> list[str]:
    tickers = list(state.get("tickers", []))
    required = [item for item in _required_evidence(state) if item != "chart"]
    if len(tickers) < 2:
        return []

    evidence = state.get("evidence", {})
    issues = []

    for evidence_type in required:
        present = []
        for ticker in tickers:
            payload = evidence.get(evidence_type, {}).get(ticker, {})
            present.append(_has_usable_payload(payload))
        if any(present) and not all(present):
            issues.append(f"asymmetric_{evidence_type}")

    return issues


def _conflict_flags(state: dict) -> list[str]:
    tickers = list(state.get("tickers", []))
    evidence = state.get("evidence", {})
    conflicts = []

    for ticker in tickers:
        market = evidence.get("market", {}).get(ticker, {})
        sentiment = evidence.get("sentiment", {}).get(ticker, {})

        trend = str(market.get("trend_label", "")).lower()
        sentiment_label = str(sentiment.get("sentiment", "")).lower()

        if trend == "upward" and sentiment_label == "negative":
            conflicts.append(f"trend_vs_sentiment:{ticker}")
        elif trend == "downward" and sentiment_label == "positive":
            conflicts.append(f"trend_vs_sentiment:{ticker}")

    return conflicts


def _follow_up_tasks(state: dict, missing: list[str], fairness_issues: list[str]) -> list[str]:
    tasks = []
    for item in missing:
        evidence_type, ticker = item.split(":", 1)
        tasks.append(f"collect_{evidence_type}:{ticker}")

    for issue in fairness_issues:
        tasks.append(f"resolve_{issue}")

    return tasks


def _confidence_label(enough_evidence: bool, missing: list[str], fairness_issues: list[str], conflicts: list[str]) -> str:
    if not enough_evidence or missing:
        return "Low"
    if fairness_issues or conflicts:
        return "Medium"
    return "High"


def run_critic_agent(state: dict) -> dict:
    next_state = clone_state(state)

    if next_state.get("needs_human"):
        next_state["critic_result"] = {
            "enough_evidence": False,
            "missing": ["human_clarification_required"],
            "fairness_issues": [],
            "conflicts": [],
            "follow_up_tasks": [],
            "confidence": "Low",
        }
        next_state["confidence"] = "Low"
        return next_state

    missing = _missing_evidence_items(next_state)
    fairness_issues = _comparison_fairness_issues(next_state)
    conflicts = _conflict_flags(next_state)

    enough_evidence = not missing and not fairness_issues and bool(next_state.get("tickers"))
    follow_up_tasks = _follow_up_tasks(next_state, missing, fairness_issues)
    confidence = _confidence_label(enough_evidence, missing, fairness_issues, conflicts)

    next_state["critic_result"] = {
        "enough_evidence": enough_evidence,
        "missing": missing,
        "fairness_issues": fairness_issues,
        "conflicts": conflicts,
        "follow_up_tasks": follow_up_tasks,
        "confidence": confidence,
    }
    next_state["confidence"] = confidence

    return next_state
