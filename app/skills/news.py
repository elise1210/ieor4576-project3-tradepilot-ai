import json
import os
import re
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from collections import Counter
from typing import Optional

from app.skills.date_utils import parse_user_date
from app.skills.finnhub_tool import finnhub_company_news, finnhub_company_news_range


COMPANY_NAMES = {
    "AAPL": "APPLE",
    "NVDA": "NVIDIA",
    "MSFT": "MICROSOFT",
    "AMZN": "AMAZON",
    "META": "META",
    "GOOGL": "ALPHABET",
    "GOOG": "ALPHABET",
    "TSLA": "TESLA",
    "AMD": "ADVANCED MICRO DEVICES",
}
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_NEWS_MODEL = os.getenv("OPENAI_NEWS_MODEL", "gpt-4o-mini")

#signal filtering
STOPWORDS = set("""
a an the and or but if to of in on for with by from is are was were be been being
this that these those it its as at into over under after before about
not no yes i you we they he she them his her our your their
have has had having do does did doing will would can could may might should
what why how when where which who today yesterday tomorrow now just more most less very
""".split())

def build_blacklist(ticker: str) -> set:
    base_blacklist = set("""
    stock stocks share shares price prices market markets investing investor investors
    etf etfs fund funds report reports says said update updates buy sell rating ratings target targets
    year years month months week weeks day days according amid expects expected
    inc company companies group maker makers
    """.split())

    dynamic = {
        ticker.lower(),
        (COMPANY_NAMES.get(ticker, "") or "").lower()
    }

    return base_blacklist.union(dynamic)


def fix_mojibake(text: str) -> str:
    if not text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def is_ticker_relevant(item: dict, ticker: str) -> bool:
    ticker = ticker.upper()
    related = (item.get("related") or "").upper()
    related_tokens = {x.strip() for x in related.split(",") if x.strip()}

    if ticker in related_tokens:
        return True

    text = " ".join([
        str(item.get("headline", "")),
        str(item.get("summary", "")),
        str(item.get("source", "")),
    ]).upper()

    return re.search(rf"\b{re.escape(ticker)}\b", text) is not None


def is_company_specific(item: dict, ticker: str) -> bool:
    if not item or "error" in item:
        return False

    ticker = ticker.upper()
    headline = (item.get("headline") or "").upper()

    if re.search(rf"\b{re.escape(ticker)}\b", headline):
        return True

    company_name = COMPANY_NAMES.get(ticker)
    if company_name and company_name in headline:
        return True

    return False


def score_price_relevance(item: dict, ticker: str) -> int:
    text = " ".join([
        str(item.get("headline", "")),
        str(item.get("summary", "")),
    ]).lower()

    high_impact = [
        "earnings", "guidance", "revenue", "profit", "miss", "beat",
        "sec", "doj", "ftc", "eu", "antitrust", "lawsuit", "ruling",
        "ban", "tariff", "sanction", "regulator", "investigation",
        "upgrade", "downgrade", "price target", "rating",
        "buyback", "repurchase", "dividend", "split",
        "iphone", "sales", "shipment", "demand", "supply", "production",
    ]

    medium_impact = [
        "forecast", "outlook", "estimate", "margin", "valuation",
        "analyst", "citi", "goldman", "morgan", "jpmorgan", "barclays",
        "etf", "buffett", "berkshire", "stake",
    ]

    score = 0

    for word in high_impact:
        if word in text:
            score += 3

    for word in medium_impact:
        if word in text:
            score += 1

    if ticker.lower() in text:
        score += 2

    return score


def filter_price_relevant_news(
    items: list,
    ticker: str,
    max_items: int = 8,
    min_score: int = 3,
) -> list:
    scored = []

    for item in items:
        if not isinstance(item, dict) or "error" in item:
            continue

        score = score_price_relevance(item, ticker)
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    strong_items = [item for score, item in scored if score >= min_score]

    if strong_items:
        return strong_items[:max_items]

    return [item for score, item in scored[:max_items]]


def wants_price_relevant_news(user_query: str) -> bool:
    query = (user_query or "").lower()

    triggers = [
        "price", "stock", "shares", "move", "moving", "why",
        "down", "up", "drop", "rally", "selloff", "spike",
        "plunge", "reaction", "impact", "catalyst", "what happened",
        "buy", "sell", "hold",
    ]

    return any(trigger in query for trigger in triggers)


def extract_news_date(user_query: str):
    return parse_user_date(user_query or "")


def filter_items_to_date(items: list, target_date: date) -> list:
    if not items:
        return []

    filtered = []

    for item in items:
        if not isinstance(item, dict) or "error" in item:
            continue

        timestamp = item.get("datetime")

        if timestamp is None:
            continue

        item_date = datetime.fromtimestamp(timestamp, timezone.utc).date()

        if item_date == target_date:
            filtered.append(item)

    return filtered


def date_signature_sentence(items: list, ticker: str, top_k: int = 6):
    text = " ".join([
        fix_mojibake(item.get("headline", "")) + " " + fix_mojibake(item.get("summary", ""))
        for item in items
    ]).lower()

    words = re.findall(r"[a-z]{3,}", text)

    blacklist = build_blacklist(ticker)

    cleaned = []

    for word in words:
        if word in STOPWORDS:
            continue
        if word in blacklist:
            continue
        cleaned.append(word)

    if not cleaned:
        return ""

    counts = Counter(cleaned)
    common = [word for word, count in counts.most_common() if count >= 2]
    top_words = common[:top_k] if common else [word for word, _ in counts.most_common(top_k)]

    return "Key topics mentioned include: " + ", ".join(top_words) + "."


def dominant_event_sentence(items: list, ticker: str) -> str:
    text = " ".join([
        fix_mojibake(item.get("headline", "")) + " " + fix_mojibake(item.get("summary", ""))
        for item in items
    ]).lower()

    rules = [
        (
            ["appeals court", "court", "ruling", "antitrust", "regulator", "doj", "ftc", "eu"],
            "legal or regulatory developments",
        ),
        (
            ["upgrade", "downgrade", "price target", "rating", "analyst"],
            "analyst views and valuation changes",
        ),
        (
            ["earnings", "guidance", "revenue", "profit", "margin", "forecast", "outlook"],
            "earnings and guidance signals",
        ),
        (
            ["iphone", "ipad", "mac", "shipments", "sales", "demand", "supply"],
            "product demand and supply signals",
        ),
        (
            ["rates", "inflation", "dollar", "recession", "sector", "etf", "nasdaq"],
            "broader macro or technology-sector sentiment",
        ),
        (
            ["buffett", "berkshire", "stake", "flows", "institutional"],
            "large-holder positioning and market flows",
        ),
    ]

    best_score = 0
    best_event = None

    for keywords, label in rules:
        score = sum(text.count(keyword) for keyword in keywords)

        if score > best_score:
            best_score = score
            best_event = label

    if best_event:
        return f"Recent {ticker} news is mainly related to {best_event}."

    return "Recent coverage is mixed with no single dominant catalyst."


def extract_daily_event_hint(items: list) -> str:
    if not items:
        return ""

    item = items[0]
    headline = fix_mojibake((item.get("headline") or "").strip())
    summary = fix_mojibake((item.get("summary") or "").strip())

    text = (headline + ". " + summary).strip()

    if not text:
        return ""

    return text[:180].rstrip(".") + "."


def summarize_news_paragraph(items: list, ticker: str, user_query: str) -> str:
    if not items:
        return f"I could not find recent company-specific news for {ticker}."

    clean_items = [
        item for item in items
        if isinstance(item, dict) and "error" not in item and item.get("datetime")
    ]

    clean_items.sort(key=lambda item: item.get("datetime", 0), reverse=True)
    clean_items = clean_items[:6]

    if not clean_items:
        return f"I could not summarize recent news for {ticker} because the feed returned no usable items."

    topic_sentence = date_signature_sentence(clean_items, ticker)
    dominant_sentence = dominant_event_sentence(clean_items, ticker)
    concrete_signal = extract_daily_event_hint(clean_items)

    if wants_price_relevant_news(user_query):
        conclusion = (
            f"Overall, these headlines provide useful context for today's {ticker} market signal, "
            "but they should not be treated as a prediction."
        )
    else:
        conclusion = (
            f"Overall, the news explains how {ticker} is being discussed recently."
        )

    parts = [
        f"Based on recent news for {ticker}:",
        topic_sentence,
        dominant_sentence,
    ]

    if concrete_signal:
        parts.append(f"Most concrete headline signal: {concrete_signal}")

    parts.append(conclusion)

    return "\n\n".join(part for part in parts if part)


def _prepare_articles_for_llm(items: list, max_items: int = 8) -> list[dict]:
    articles = []

    for item in items[:max_items]:
        if not isinstance(item, dict) or "error" in item:
            continue

        timestamp = item.get("datetime")
        date_str = (
            datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if timestamp
            else item.get("date", "N/A")
        )

        articles.append({
            "date": date_str,
            "headline": fix_mojibake(item.get("headline", ""))[:240],
            "summary": fix_mojibake(item.get("summary", ""))[:700],
            "source": item.get("source", ""),
        })

    return articles


def summarize_news_with_openai(
    items: list,
    ticker: str,
    user_query: str,
    model: Optional[str] = None,
    timeout: int = 12,
) -> Optional[str]:
    """
    Optional AI news summary.

    Returns None when OPENAI_API_KEY is missing or when the API call fails, so
    callers can safely fall back to the deterministic rule-based summary.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    articles = _prepare_articles_for_llm(items)
    if not articles:
        return None

    payload = {
        "model": model or os.getenv("OPENAI_NEWS_MODEL", OPENAI_NEWS_MODEL),
        "temperature": 0.2,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize company news for an educational finance app. "
                    "Use only the provided articles. Do not invent facts. "
                    "Do not predict stock prices or give investment advice. "
                    "If evidence is thin, say so briefly."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticker": ticker,
                        "user_query": user_query,
                        "articles": articles,
                        "task": (
                            "Write a concise, natural-language summary in 2-4 sentences. "
                            "Mention the main event and uncertainty if relevant."
                        ),
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        summary = data["choices"][0]["message"]["content"].strip()
        return summary or None
    except Exception:
        return None


def summarize_news_with_openai_result(
    items: list,
    ticker: str,
    user_query: str,
    model: Optional[str] = None,
    timeout: int = 12,
) -> dict:
    """
    Optional AI news summary with debug metadata.

    This keeps OpenAI failures non-fatal while making the fallback reason
    visible to the API/frontend.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "summary": None,
            "summary_source": "rule_based",
            "summary_fallback_used": True,
            "summary_error": "OPENAI_API_KEY is not configured.",
        }

    articles = _prepare_articles_for_llm(items)
    if not articles:
        return {
            "summary": None,
            "summary_source": "rule_based",
            "summary_fallback_used": True,
            "summary_error": "No usable articles available for OpenAI summary.",
        }

    payload = {
        "model": model or os.getenv("OPENAI_NEWS_MODEL", OPENAI_NEWS_MODEL),
        "temperature": 0.2,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize company news for an educational finance app. "
                    "Use only the provided articles. Do not invent facts. "
                    "Do not predict stock prices or give investment advice. "
                    "If evidence is thin, say so briefly."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticker": ticker,
                        "user_query": user_query,
                        "articles": articles,
                        "task": (
                            "Write a concise, natural-language summary in 2-4 sentences. "
                            "Mention the main event and uncertainty if relevant."
                        ),
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    }

    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        summary = data["choices"][0]["message"]["content"].strip()
        if not summary:
            return {
                "summary": None,
                "summary_source": "rule_based",
                "summary_fallback_used": True,
                "summary_error": "OpenAI returned an empty summary.",
            }

        return {
            "summary": summary,
            "summary_source": "openai",
            "summary_fallback_used": False,
            "summary_error": None,
        }
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8")[:400]
        except Exception:
            detail = str(e)
        return {
            "summary": None,
            "summary_source": "rule_based",
            "summary_fallback_used": True,
            "summary_error": f"OpenAI API HTTP error: {e.code}. {detail}",
        }
    except Exception as e:
        return {
            "summary": None,
            "summary_source": "rule_based",
            "summary_fallback_used": True,
            "summary_error": f"OpenAI summary failed: {str(e)}",
        }


def format_news_items(items: list) -> list:
    formatted = []

    for item in items:
        timestamp = item.get("datetime")
        date_str = (
            datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            if timestamp
            else "N/A"
        )

        formatted.append({
            "date": date_str,
            "headline": fix_mojibake(item.get("headline", "")),
            "summary": fix_mojibake(item.get("summary", "")),
            "source": item.get("source", ""),
            "url": item.get("url", ""),
        })

    return formatted


def run_news_skill(
    ticker: str,
    user_query: str = "",
    target_date=None,
    days: int = 7,
    max_items: int = 8,
    query: Optional[str] = None,
) -> dict:
    """
    News skill for TradePilot AI.

    This skill:
    1. Retrieves recent company news from Finnhub.
    2. Filters for ticker/company relevance.
    3. Prioritizes price-relevant news when the user asks about buy/hold/sell or movement.
    4. Produces a compact news summary for downstream decision-making.

    It does NOT predict future stock prices.
    """

    ticker = ticker.upper().strip()
    if query is not None:
        user_query = query

    asked_date = parse_user_date(str(target_date)) if target_date is not None else extract_news_date(user_query)

    if asked_date:
        start_date = asked_date
        end_date = asked_date
        lookback_days = 1
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        lookback_days = days

    raw_items = finnhub_company_news(
        ticker=ticker,
        days=lookback_days,
        max_items=30,
    ) if not asked_date else finnhub_company_news_range(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        max_items=30,
    )

    usable_items = [
        item for item in raw_items
        if isinstance(item, dict) and "error" not in item
    ]

    if asked_date:
        usable_items = filter_items_to_date(usable_items, asked_date)

    relevant_items = [
        item for item in usable_items
        if is_ticker_relevant(item, ticker) and is_company_specific(item, ticker)
    ]

    if wants_price_relevant_news(user_query):
        selected_items = filter_price_relevant_news(
            relevant_items,
            ticker=ticker,
            max_items=max_items,
            min_score=1,
        )
    else:
        selected_items = relevant_items[:max_items]

    rule_based_summary = summarize_news_paragraph(
        selected_items,
        ticker=ticker,
        user_query=user_query,
    )
    ai_summary_result = summarize_news_with_openai_result(
        selected_items,
        ticker=ticker,
        user_query=user_query,
    )
    ai_summary = ai_summary_result["summary"]
    summary = ai_summary or rule_based_summary

    return {
        "ticker": ticker,
        "source": "Finnhub company news",
        "requested_date": str(asked_date) if asked_date else None,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "article_count": len(selected_items),
        "summary": summary,
        "summary_source": ai_summary_result["summary_source"],
        "summary_fallback_used": ai_summary_result["summary_fallback_used"],
        "summary_error": ai_summary_result["summary_error"],
        "items": format_news_items(selected_items),
    }


__all__ = [
    "extract_news_date",
    "run_news_skill",
    "summarize_news_with_openai",
    "summarize_news_with_openai_result",
]
