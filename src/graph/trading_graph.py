"""
trading_graph.py

Complete LangGraph implementation of the hackathon trading agent.

Graph flow:
START
  -> get_account
  -> get_market_data
  -> get_news
  -> llm_reasoning
  -> risk_check
  -> execute_order
  -> write_journal
  -> END

Each node updates a shared TradingState. This makes the perception -> reasoning
-> action -> journal loop explicit, inspectable, and demo-friendly.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agent.llm_reasoner import LLMReasoningAgent
from src.tools.broker import BrokerTool
from src.tools.journal import TradeJournal
from src.tools.market_data import MarketDataTool
from src.tools.news_tool import NewsTool


load_dotenv()


class TradingState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    symbol: str
    execute_orders: bool
    demo_trade_mode: bool

    account_summary: dict[str, Any]
    price_snapshot: dict[str, Any]
    news_data: dict[str, Any]
    news_signal: dict[str, Any]
    decision: dict[str, Any]
    risk_check: dict[str, Any]
    order_result: dict[str, Any]
    journal_entry: dict[str, Any]

    errors: list[str]


class TradingGraph:
    """LangGraph-based autonomous trading workflow."""

    def __init__(
        self,
        execute_orders: bool = False,
        demo_trade_mode: bool = False,
        max_position_value_pct: float = 0.10,
        confidence_threshold: float = 0.70,
        demo_quantity: int = 1,
    ) -> None:
        self.execute_orders = execute_orders
        self.demo_trade_mode = demo_trade_mode
        self.max_position_value_pct = max_position_value_pct
        self.confidence_threshold = confidence_threshold
        self.demo_quantity = demo_quantity

        self.broker = BrokerTool()
        self.market_data = MarketDataTool()
        self.news_tool = NewsTool()
        self.llm_reasoner = LLMReasoningAgent()
        self.journal = TradeJournal()

        self.graph = self._build_graph()

    @staticmethod
    def _append_error(state: TradingState, message: str) -> None:
        state.setdefault("errors", [])
        state["errors"].append(message)

    def get_account_node(self, state: TradingState) -> TradingState:
        account_summary = self.broker.get_account_summary()
        state["account_summary"] = account_summary
        if account_summary.get("error"):
            self._append_error(state, account_summary.get("message", "Account tool failed."))
        return state

    def get_market_data_node(self, state: TradingState) -> TradingState:
        symbol = state["symbol"]
        price_snapshot = self.market_data.get_price_snapshot(symbol)
        state["price_snapshot"] = price_snapshot
        if price_snapshot.get("error"):
            self._append_error(state, price_snapshot.get("message", "Market data tool failed."))
        return state

    def get_news_node(self, state: TradingState) -> TradingState:
        symbol = state["symbol"]

        news_data = self.news_tool.get_recent_news(symbol=symbol, hours_back=168, limit=5)
        if news_data.get("article_count", 0) == 0:
            news_data = self.news_tool.get_yfinance_news(symbol=symbol, limit=5)

        news_signal = self.news_tool.summarize_news_signal(news_data)

        state["news_data"] = news_data
        state["news_signal"] = news_signal

        if news_data.get("error"):
            self._append_error(state, news_data.get("message", "News tool failed."))
        return state

    def llm_reasoning_node(self, state: TradingState) -> TradingState:
        decision = self.llm_reasoner.decide(
            symbol=state["symbol"],
            price_snapshot=state.get("price_snapshot", {}),
            news_signal=state.get("news_signal", {}),
            account_summary=state.get("account_summary", {}),
        )

        # Optional hackathon demo mode: only for Alpaca paper trading. It allows
        # a tiny demonstrational BUY when the evidence is directionally aligned
        # but the LLM remains overly cautious. The journal still records that
        # this was a demo override.
        if (
            self.demo_trade_mode
            and self.execute_orders
            and decision.get("action") == "HOLD"
            and state.get("price_snapshot", {}).get("trend") == "up"
            and state.get("news_signal", {}).get("signal") == "positive"
        ):
            decision = {
                **decision,
                "original_action": "HOLD",
                "action": "BUY",
                "confidence": max(float(decision.get("confidence", 0.0)), 0.71),
                "decision_source": "demo_trade_override_after_llm_hold",
                "rationale": (
                    "Demo trade mode is enabled for paper trading. The LLM preferred HOLD, "
                    "but price trend is up and news signal is positive, so the graph permits "
                    "a tiny 1-share paper BUY to demonstrate broker execution. This is not "
                    "financial advice and is only for the hackathon demo. Original rationale: "
                    + str(decision.get("rationale", ""))
                ),
            }

        state["decision"] = decision
        return state

    def risk_check_node(self, state: TradingState) -> TradingState:
        decision = state.get("decision", {"action": "HOLD", "confidence": 0.0})
        price_snapshot = state.get("price_snapshot", {})
        account_summary = state.get("account_summary", {})

        action = str(decision.get("action", "HOLD")).upper()
        confidence = float(decision.get("confidence", 0.0))

        if account_summary.get("error"):
            risk = {"approved": False, "quantity": 0, "reason": "Account data unavailable."}
        elif account_summary.get("trading_blocked") or account_summary.get("account_blocked"):
            risk = {"approved": False, "quantity": 0, "reason": "Trading/account is blocked by broker."}
        elif action in {"HOLD", "SKIP"}:
            risk = {"approved": False, "quantity": 0, "reason": f"No order required for action {action}."}
        elif confidence < self.confidence_threshold:
            risk = {
                "approved": False,
                "quantity": 0,
                "reason": f"Confidence {confidence:.2f} is below threshold {self.confidence_threshold:.2f}.",
            }
        else:
            latest_price = price_snapshot.get("latest_price")
            if not latest_price or latest_price <= 0:
                risk = {"approved": False, "quantity": 0, "reason": "Invalid or missing latest price."}
            else:
                if self.demo_trade_mode and self.execute_orders:
                    quantity = self.demo_quantity
                    max_order_value = float(latest_price) * quantity
                    reason = "Demo trade mode approved a tiny paper order."
                else:
                    portfolio_value = float(account_summary.get("portfolio_value", 0.0))
                    max_order_value = portfolio_value * self.max_position_value_pct
                    quantity = int(max_order_value // float(latest_price))
                    reason = f"Risk check approved. Order value capped at {self.max_position_value_pct:.0%} of portfolio."

                if quantity <= 0:
                    risk = {"approved": False, "quantity": 0, "reason": "Portfolio allocation too small to buy one share."}
                else:
                    risk = {
                        "approved": True,
                        "quantity": quantity,
                        "max_order_value": max_order_value,
                        "reason": reason,
                    }

        state["risk_check"] = risk
        return state

    def execute_order_node(self, state: TradingState) -> TradingState:
        decision = state.get("decision", {"action": "HOLD"})
        risk = state.get("risk_check", {"approved": False})
        symbol = state["symbol"]

        order_result: dict[str, Any] = {
            "executed": False,
            "paper_trading_enabled": self.execute_orders,
            "demo_trade_mode": self.demo_trade_mode,
            "reason": "Dry-run mode" if not self.execute_orders else "Order not approved.",
        }

        if self.execute_orders and risk.get("approved"):
            order_result = self.broker.place_market_order(
                symbol=symbol,
                side=str(decision["action"]).lower(),
                quantity=float(risk["quantity"]),
            )
            order_result["paper_trading_enabled"] = self.execute_orders
            order_result["demo_trade_mode"] = self.demo_trade_mode
        elif risk.get("approved"):
            order_result = {
                "executed": False,
                "paper_trading_enabled": self.execute_orders,
                "demo_trade_mode": self.demo_trade_mode,
                "reason": "Dry-run mode: order approved but not submitted.",
                "planned_order": {
                    "symbol": symbol,
                    "side": decision["action"],
                    "quantity": risk["quantity"],
                },
            }

        state["order_result"] = order_result
        return state

    def journal_node(self, state: TradingState) -> TradingState:
        decision = state.get("decision", {"action": "HOLD", "confidence": 0.0, "rationale": "No decision."})
        price_snapshot = state.get("price_snapshot", {})

        error_notes = None
        if state.get("errors"):
            error_notes = " | ".join(state["errors"])
        elif price_snapshot.get("error"):
            error_notes = price_snapshot.get("message")

        entry = self.journal.log_decision(
            symbol=state["symbol"],
            action=str(decision.get("action", "HOLD")),
            confidence=float(decision.get("confidence", 0.0)),
            rationale=str(decision.get("rationale", "")),
            price_data=price_snapshot,
            news_data={
                "raw_news": state.get("news_data", {}),
                "news_signal": state.get("news_signal", {}),
            },
            risk_check=state.get("risk_check", {}),
            order_result=state.get("order_result", {}),
            portfolio_state=state.get("account_summary", {}),
            error_notes=error_notes,
        )

        state["journal_entry"] = entry
        return state

    def _build_graph(self):
        builder = StateGraph(TradingState)

        builder.add_node("get_account", self.get_account_node)
        builder.add_node("get_market_data", self.get_market_data_node)
        builder.add_node("get_news", self.get_news_node)
        builder.add_node("llm_reasoning", self.llm_reasoning_node)
        builder.add_node("risk_check", self.risk_check_node)
        builder.add_node("execute_order", self.execute_order_node)
        builder.add_node("write_journal", self.journal_node)

        builder.set_entry_point("get_account")
        builder.add_edge("get_account", "get_market_data")
        builder.add_edge("get_market_data", "get_news")
        builder.add_edge("get_news", "llm_reasoning")
        builder.add_edge("llm_reasoning", "risk_check")
        builder.add_edge("risk_check", "execute_order")
        builder.add_edge("execute_order", "write_journal")
        builder.add_edge("write_journal", END)

        return builder.compile()

    def run_symbol(self, symbol: str) -> TradingState:
        initial_state: TradingState = {
            "symbol": symbol.upper().strip(),
            "execute_orders": self.execute_orders,
            "demo_trade_mode": self.demo_trade_mode,
            "errors": [],
        }
        return self.graph.invoke(initial_state)
