# TradePilot AI  
IEOR 4576 Project 3 — Personal Stock Decision Agent

## Overview

TradePilot AI is a **multi-agent financial decision system** that aggregates stock news, analyzes sentiment, integrates market signals, and produces actionable **BUY / HOLD / SELL** recommendations.

Instead of manually checking multiple platforms (Yahoo Finance, Bloomberg, charts), users can simply ask:

> “Should I buy Apple today?”

The system returns a structured, decision-oriented response.


## Target Users

- Retail investors (age 25–40)
- Users of Robinhood / Webull
- Limited time to analyze financial information
- Lack experience interpreting market signals


##  Problem

Users today must:

- Search news across multiple platforms
- Interpret sentiment manually
- Analyze price charts separately

Core issue:

**Users cannot translate fragmented information into a clear decision.**


## Solution

TradePilot AI combines:

- News signals
- Sentiment analysis
- Market trends
- Company context

→ into a **single, structured decision output**


## System Architecture

The system follows a **multi-agent architecture**:

User Query  
↓  
main.py (orchestrator)  
↓  
[News Agent] → [Sentiment Agent]  
↓  
[Market Agent] → [Company Agent]  
↓  
[Decision Agent]  
↓  
Final Recommendation  

## Project Structure

```bash
tradepilot-ai/
├── app/
│   ├── main.py                  # FastAPI entry point (orchestrator)
│
│   ├── agents/                  # Multi-agent system
│   │   ├── news_agent.py        # Fetch, filter, and summarize financial news
│   │   ├── market_agent.py      # Price trend & volatility analysis (7-day signals)
│   │   ├── company_agent.py     # Company background & key context
│   │   └── decision_agent.py    # Final BUY / HOLD / SELL decision logic
│
│   ├── tools/                   # External data + model integrations
│   │   ├── finnhub_tool.py      # Financial news API (Finnhub)
│   │   ├── yfinance_tool.py     # Market data (price, history)
│   │   ├── sentiment_tool.py    # FinBERT sentiment model
│   │   └── chart_tool.py        # Price trend visualization
│
│   ├── prompts.py               # Prompt templates (optional / LLM usage)
│   └── config.py                # Configuration & constants
│
├── frontend/
│   └── index.html               # Simple user interface
│
├── README.md                    # Project documentation
├── Dockerfile                   # Containerization
├── cloudbuild.yaml              # Cloud Run deployment config
└── business_one_pager.md        # Business proposal document
```


## Technical Design

### Multi-Agent System
- Modular agents handle different tasks
- Separation of concerns improves scalability

### Tool Calling
- Finnhub API → financial news
- yfinance → market data
- FinBERT → sentiment analysis

### Context Engineering
- Converts raw data into structured signals:
  - sentiment score
  - trend
  - volatility
- Enables consistent decision logic

### Decision Logic
- Weighted combination of:
  - sentiment signal
  - market trend
- Outputs interpretable reasoning


## Example Output
Recommendation: **HOLD**

Reason:

News sentiment: Neutral (+0.05)  
Price trend: Downward (-1.2% past 7 days)  
Risk level: Low  
Key driver: Mixed earnings outlook  


## Business Model

- Free tier:
  - News summary
  - Basic stock info

- Pro tier ($10/month):
  - Sentiment signals
  - Buy/sell recommendations
  - Alerts


## Unit Economics (Estimate)

- ~5 queries/day/user  
- ~1k tokens/query  
- Cost ≈ $0.002/query  

→ Monthly cost ≈ $0.30  
→ Revenue ≈ $10  

High margin (~97%)

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload  #example
```

## Deployment

- Backend: FastAPI
- Containerization: Docker
- Deployment: Cloud Run
- Public URL: (add after deploy)

## Course Concepts Used

This project incorporates key concepts from IEOR 4576:

- Multi-Agent Systems (modular agents + orchestrator)
- Tool Calling (external APIs for real-time data)
- Context Engineering (structured signals for reasoning)
- LLM-style Reasoning (decision synthesis layer)
- Evaluation (rule-based signal scoring)

## ⚠️ Disclaimer

This system provides informational signals only and does not predict future stock prices or provide financial advice.