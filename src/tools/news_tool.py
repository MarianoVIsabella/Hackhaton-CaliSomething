"""
news_tool.py

Financial news retrieval tool.

This tool uses Alpaca News API first. If news retrieval fails, it returns a
safe error object instead of crashing the trading agent.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any


from dotenv import load_dotenv
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
import yfinance as yf


load_dotenv()


class NewsTool:
    """
    Retrieves recent financial news for monitored tickers.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca API keys. Please set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file."
            )

        self.client = NewsClient(api_key, secret_key)

    def get_recent_news(
        self,
        symbol: str,
        hours_back: int = 168,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve recent news for one stock symbol.
        """
        cleaned_symbol = symbol.upper().strip()

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            request = NewsRequest(
                symbols=cleaned_symbol,
                start=start_time,
                end=end_time,
                limit=limit,
            )

            news_response = self.client.get_news(request)

            articles = []

            for article in news_response.data.get(cleaned_symbol, []):
                articles.append(
                    {
                        "headline": article.headline,
                        "summary": article.summary,
                        "author": article.author,
                        "created_at": str(article.created_at),
                        "updated_at": str(article.updated_at),
                        "url": article.url,
                        "source": article.source,
                    }
                )

            return {
                "error": False,
                "symbol": cleaned_symbol,
                "article_count": len(articles),
                "articles": articles,
                "source": "alpaca_news_api",
            }

        except Exception as error:
            return {
                "error": True,
                "symbol": cleaned_symbol,
                "message": f"Failed to retrieve news: {error}",
                "source": "alpaca_news_api",
            }

    def get_yfinance_news(self, symbol: str, limit: int = 5) -> dict[str, Any]:
        cleaned_symbol = symbol.upper().strip()

        try:
            ticker = yf.Ticker(cleaned_symbol)
            raw_news = ticker.news or []

            articles = []

            for item in raw_news[:limit]:
                content = item.get("content", item)

                articles.append(
                    {
                        "headline": content.get("title", ""),
                        "summary": content.get("summary", ""),
                        "author": content.get("provider", {}).get("displayName", ""),
                        "created_at": str(content.get("pubDate", "")),
                        "updated_at": "",
                        "url": content.get("canonicalUrl", {}).get("url", ""),
                        "source": "yfinance_news",
                    }
                )

            return {
                "error": False,
                "symbol": cleaned_symbol,
                "article_count": len(articles),
                "articles": articles,
                "source": "yfinance_news",
            }

        except Exception as error:
            return {
                "error": True,
                "symbol": cleaned_symbol,
                "message": f"Failed to retrieve yfinance news: {error}",
                "source": "yfinance_news",
            }
    
    def summarize_news_signal(self, news_data: dict[str, Any]) -> dict[str, Any]:
        """
        Very simple non-LLM news signal.

        Later, this can be replaced by an LLM reasoning agent.
        """
        if news_data.get("error"):
            return {
                "signal": "unknown",
                "confidence": 0.0,
                "reason": news_data.get("message", "News unavailable."),
            }

        articles = news_data.get("articles", [])

        if not articles:
            return {
                "signal": "neutral",
                "confidence": 0.4,
                "reason": "No recent news found for the ticker.",
            }

        positive_words = {
            "beats",
            "growth",
            "upgrade",
            "surge",
            "profit",
            "record",
            "strong",
            "bullish",
            "expands",
            "gain",
        }

        negative_words = {
            "misses",
            "downgrade",
            "lawsuit",
            "fall",
            "decline",
            "loss",
            "weak",
            "bearish",
            "investigation",
            "risk",
        }

        text = " ".join(
            [
                f"{article.get('headline', '')} {article.get('summary', '')}"
                for article in articles
            ]
        ).lower()

        positive_hits = sum(1 for word in positive_words if word in text)
        negative_hits = sum(1 for word in negative_words if word in text)

        if positive_hits > negative_hits:
            signal = "positive"
            confidence = min(0.75, 0.45 + positive_hits * 0.1)
        elif negative_hits > positive_hits:
            signal = "negative"
            confidence = min(0.75, 0.45 + negative_hits * 0.1)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "positive_hits": positive_hits,
            "negative_hits": negative_hits,
            "reason": (
                f"Analyzed {len(articles)} recent article(s). "
                f"Positive keyword hits: {positive_hits}. "
                f"Negative keyword hits: {negative_hits}."
            ),
        }


if __name__ == "__main__":
    news_tool = NewsTool()

    for ticker in ["AAPL", "MSFT", "NVDA"]:
        print(f"\nNews for {ticker}:")
        news = news_tool.get_recent_news(ticker)

        if news.get("article_count", 0) == 0:
            news = news_tool.get_yfinance_news(ticker)
        print(news)

        print("\nNews signal:")
        print(news_tool.summarize_news_signal(news))