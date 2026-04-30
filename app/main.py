from dotenv import load_dotenv
import os

load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.news_agent import run_news_agent
from app.tools.sentiment_tool import analyze_news_sentiment
from app.agents.decision_agent import generate_decision, format_decision_output
from app.agents.market_agent import run_market_agent


app = FastAPI(title="TradePilot AI")


class ChatRequest(BaseModel):
    query: str
    ticker: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    ticker = req.ticker.upper().strip()
    query = req.query.strip()

    print(f"Processing: {ticker} | {query}")

    news_result = run_news_agent(
        ticker=ticker,
        user_query=query,
        days=7,
        max_items=8,
    )

    try:
        sentiment_result = analyze_news_sentiment(news_result)
    except Exception as e:
        print("Sentiment error:", e)
        sentiment_result = {
            "sentiment": "neutral",
            "score": 0.0,
            "dispersion": 0.0
        }

    market_result = run_market_agent(ticker)

    decision = generate_decision(
        ticker=ticker,
        news_result=news_result,
        sentiment_result=sentiment_result,
        market_result=market_result,
        company_result=None,
    )

    return {
        "ticker": ticker,
        "query": query,
        "answer": format_decision_output(decision),
    }