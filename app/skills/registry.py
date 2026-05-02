from app.skills.chart import run_chart_skill
from app.skills.fundamentals import run_fundamentals_skill
from app.skills.market import run_market_skill
from app.skills.news import run_news_skill
from app.skills.sentiment import run_sentiment_skill


REAL_SKILLS = {
    "news": run_news_skill,
    "market": run_market_skill,
    "fundamentals": run_fundamentals_skill,
    "sentiment": run_sentiment_skill,
    "chart": run_chart_skill,
}


__all__ = ["REAL_SKILLS"]
