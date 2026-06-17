
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


load_dotenv()


@dataclass
class BrokerConfig:
    api_key: str
    secret_key: str
    paper: bool = True


class BrokerTool:
    """
    Safe wrapper around Alpaca TradingClient.
    """

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca API keys. Please set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file."
            )

        self.config = BrokerConfig(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
        )

        self.client = TradingClient(
            api_key=self.config.api_key,
            secret_key=self.config.secret_key,
            paper=self.config.paper,
        )

    def get_account_summary(self) -> dict[str, Any]:
        """
        Return useful account information.
        """
        try:
            account = self.client.get_account()

            return {
                "status": str(account.status),
                "currency": account.currency,
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": bool(account.pattern_day_trader),
                "trading_blocked": bool(account.trading_blocked),
                "account_blocked": bool(account.account_blocked),
            }

        except Exception as error:
            return {
                "error": True,
                "message": f"Failed to retrieve account summary: {error}",
            }

    def get_positions(self) -> list[dict[str, Any]]:
        """
        Return all open positions.
        """
        try:
            positions = self.client.get_all_positions()

            return [
                {
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "market_value": float(position.market_value),
                    "average_entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                }
                for position in positions
            ]

        except Exception as error:
            return [
                {
                    "error": True,
                    "message": f"Failed to retrieve positions: {error}",
                }
            ]

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> dict[str, Any]:
        """
        Submit a market order.

        side must be:
        - "buy"
        - "sell"
        """
        try:
            normalized_side = side.lower().strip()

            if normalized_side not in {"buy", "sell"}:
                return {
                    "error": True,
                    "message": "Invalid order side. Use 'buy' or 'sell'.",
                }

            if quantity <= 0:
                return {
                    "error": True,
                    "message": "Quantity must be greater than zero.",
                }

            order_side = (
                OrderSide.BUY
                if normalized_side == "buy"
                else OrderSide.SELL
            )

            order_request = MarketOrderRequest(
                symbol=symbol.upper().strip(),
                qty=quantity,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )

            order = self.client.submit_order(order_data=order_request)

            return {
                "error": False,
                "order_id": str(order.id),
                "symbol": order.symbol,
                "side": str(order.side),
                "quantity": float(order.qty),
                "status": str(order.status),
                "submitted_at": str(order.submitted_at),
            }

        except Exception as error:
            return {
                "error": True,
                "message": f"Failed to place market order: {error}",
            }


if __name__ == "__main__":
    broker = BrokerTool()

    print("Account summary:")
    print(broker.get_account_summary())

    print("\nOpen positions:")
    print(broker.get_positions())