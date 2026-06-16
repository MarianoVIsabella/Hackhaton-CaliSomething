"""
market_data.py

Market data wrapper for the trading agent.

This module retrieves real market prices from Alpaca Market Data.
The agent must never invent prices; every price must come from this tool.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import yfinance as yf


load_dotenv()


class MarketDataTool:
    """
    Safe wrapper around Alpaca market data.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca API keys. Please set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file."
            )

        self.client = StockHistoricalDataClient(api_key, secret_key)

    def get_yfinance_bars(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "5m",
    ) -> dict[str, Any]:
        """
        Fallback market bars from yfinance.

        Used when Alpaca recent bars are unavailable.
        """
        cleaned_symbol = symbol.upper().strip()

        try:
            ticker = yf.Ticker(cleaned_symbol)
            history = ticker.history(period=period, interval=interval)

            if history.empty:
                return {
                    "error": True,
                    "symbol": cleaned_symbol,
                    "message": "yfinance returned no bar data.",
                    "source": "yfinance_bars",
                }

            parsed_bars = []

            for timestamp, row in history.iterrows():
                parsed_bars.append(
                    {
                        "timestamp": str(timestamp),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": int(row["Volume"]),
                    }
                )

            return {
                "error": False,
                "symbol": cleaned_symbol,
                "bar_count": len(parsed_bars),
                "bars": parsed_bars,
                "source": "yfinance_bars",
            }

        except Exception as error:
            return {
                "error": True,
                "symbol": cleaned_symbol,
                "message": f"Failed to retrieve yfinance bars: {error}",
                "source": "yfinance_bars",
            }
    
    def get_latest_trade(self, symbol: str) -> dict[str, Any]:
        """
        Get the latest available trade price for a stock symbol.
        """
        try:
            cleaned_symbol = symbol.upper().strip()

            request = StockLatestTradeRequest(symbol_or_symbols=cleaned_symbol)
            response = self.client.get_stock_latest_trade(request)

            trade = response[cleaned_symbol]

            return {
                "error": False,
                "symbol": cleaned_symbol,
                "price": float(trade.price),
                "size": int(trade.size),
                "timestamp": str(trade.timestamp),
                "source": "alpaca_latest_trade",
            }

        except Exception as error:
            return {
                "error": True,
                "symbol": symbol.upper().strip(),
                "message": f"Failed to retrieve latest trade: {error}",
            }

    def get_recent_bars(
        self,
        symbol: str,
        minutes: int = 30,
    ) -> dict[str, Any]:
        """
        Get recent minute bars for a stock symbol.
        """
        try:
            cleaned_symbol = symbol.upper().strip()

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=minutes)

            request = StockBarsRequest(
                symbol_or_symbols=cleaned_symbol,
                timeframe=TimeFrame.Minute,
                start=start_time,
                end=end_time,
            )

            bars = self.client.get_stock_bars(request)

            symbol_bars = bars.data.get(cleaned_symbol, [])

            parsed_bars = [
                {
                    "timestamp": str(bar.timestamp),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
                for bar in symbol_bars
            ]

            return {
                "error": False,
                "symbol": cleaned_symbol,
                "bar_count": len(parsed_bars),
                "bars": parsed_bars,
                "source": "alpaca_stock_bars",
            }

        except Exception as error:
            return {
                "error": True,
                "symbol": symbol.upper().strip(),
                "message": f"Failed to retrieve recent bars: {error}",
            }

   
    def get_price_snapshot(self, symbol: str) -> dict[str, Any]:
        """
        Get a compact price snapshot for decision-making.

        Uses Alpaca latest trade first.
        Uses Alpaca recent bars first.
        Falls back to yfinance bars if Alpaca bars are unavailable.
        """
        latest_trade = self.get_latest_trade(symbol)

        if latest_trade.get("error"):
            return latest_trade

        recent_bars = self.get_recent_bars(symbol, minutes=30)

        if recent_bars.get("error") or recent_bars.get("bar_count", 0) == 0:
            recent_bars = self.get_yfinance_bars(symbol, period="1d", interval="5m")

        if recent_bars.get("error") or recent_bars.get("bar_count", 0) == 0:
            return {
                "error": False,
                "symbol": latest_trade["symbol"],
                "latest_price": latest_trade["price"],
                "timestamp": latest_trade["timestamp"],
                "trend": "unknown",
                "bar_count": 0,
                "source": "alpaca_latest_trade_only",
                "warning": "Recent bars unavailable from Alpaca and yfinance; trend not computed.",
            }

        bars = recent_bars["bars"]
        first_close = bars[0]["close"]
        last_close = bars[-1]["close"]

        price_change = last_close - first_close
        price_change_pct = (price_change / first_close) * 100 if first_close else 0.0

        if price_change_pct > 0.2:
            trend = "up"
        elif price_change_pct < -0.2:
            trend = "down"
        else:
            trend = "flat"

        return {
            "error": False,
            "symbol": latest_trade["symbol"],
            "latest_price": latest_trade["price"],
            "timestamp": latest_trade["timestamp"],
            "bar_count": len(bars),
            "first_close": first_close,
            "last_close": last_close,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "trend": trend,
            "latest_trade_source": latest_trade["source"],
            "bars_source": recent_bars["source"],
            "source": "alpaca_latest_trade_with_bar_fallback",
        }
if __name__ == "__main__":
    market_data = MarketDataTool()

    for ticker in ["AAPL", "MSFT", "NVDA"]:
        print(f"\nPrice snapshot for {ticker}:")
        print(market_data.get_price_snapshot(ticker))