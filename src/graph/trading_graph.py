
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
from src.agent.judge_agent import JudgeAgent
from src.tools.feedback_memory import FeedbackMemory


load_dotenv()


SECTOR_MAP: dict[str, str] = {
    # Technology / AI
    "AAPL": "technology",
    "MSFT": "technology",
    "GOOGL": "technology",
    "GOOG": "technology",
    "AMZN": "technology",
    "META": "technology",
    "NVDA": "semiconductor",
    "AMD": "semiconductor",
    "INTC": "semiconductor",

    # Electric vehicles / mobility
    "TSLA": "electric_vehicle",
    "RIVN": "electric_vehicle",
    "LCID": "electric_vehicle",

    # Oil, gas, and energy
    "XOM": "oil_gas",
    "CVX": "oil_gas",
    "OXY": "oil_gas",
    "BP": "oil_gas",
    "SHEL": "oil_gas",
    "COP": "oil_gas",
    "SLB": "oil_gas",
    "HAL": "oil_gas",
}

TARGET_SECTOR_ALIASES: dict[str, set[str]] = {
    "oil_gas": {"oil_gas", "energy", "petroleum", "oil", "gas"},
}


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
    positions: list[dict[str, Any]]
    current_position: dict[str, Any]
    pre_judge_decision: dict[str, Any]
    feedback_context: dict[str, Any]
    feedback_policy: dict[str, Any]
    feedback_effects: list[str]

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
        self.judge_agent = JudgeAgent()
        self.feedback_memory = FeedbackMemory()

        self.graph = self._build_graph()

    @staticmethod
    def _append_error(state: TradingState, message: str) -> None:
        state.setdefault("errors", [])
        state["errors"].append(message)

    def get_account_node(self, state: TradingState) -> TradingState:
        account_summary = self.broker.get_account_summary()
        positions = self.broker.get_positions()

        symbol = state["symbol"]

        current_position = next(
            (
                position
                for position in positions
                if position.get("symbol") == symbol
            ),
            {
                "symbol": symbol,
                "quantity": 0,
                "market_value": 0,
                "unrealized_pl": 0,
                "unrealized_plpc": 0,
            },
        )

        state["account_summary"] = account_summary
        state["positions"] = positions
        state["current_position"] = current_position

        return state
    
    def read_feedback_node(self, state: TradingState) -> TradingState:
        """Load latest user feedback and convert it into policy.

        Feedback is useful only if it becomes state. This node makes the
        feedback visible to the LLM, the Judge, and the deterministic policy
        layer.
        """
        feedback_context = self.feedback_memory.read_latest_summary(limit=10)
        state["feedback_context"] = feedback_context
        state["feedback_policy"] = feedback_context.get("policy", {}) or {}
        state["feedback_effects"] = []
        return state

    def judge_decision_node(self, state: TradingState) -> TradingState:
        """Review the LLM decision.

        The Judge may accept or change the LLM decision. Feedback policy is not
        enforced here; it is enforced by the next deterministic node so that
        user instructions work even if the Judge ignores them or the LLM fails.
        """
        proposed_decision = state["decision"]

        reviewed_decision = self.judge_agent.review(
            symbol=state["symbol"],
            price_snapshot=state.get("price_snapshot", {}),
            news_signal=state.get("news_signal", {}),
            account_summary=state.get("account_summary", {}),
            current_position=state.get("current_position", {}),
            proposed_decision=proposed_decision,
            feedback_context=state.get("feedback_context", {}),
        )

        state["pre_judge_decision"] = proposed_decision
        state["decision"] = reviewed_decision

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
            current_position=state.get("current_position", {}),
            feedback_context=state.get("feedback_context", {})
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

    @staticmethod
    def _owned_quantity(position: dict[str, Any]) -> float:
        return float(position.get("quantity", position.get("qty", 0)) or 0)

    def _symbol_matches_feedback_targets(self, symbol: str, policy: dict[str, Any]) -> tuple[bool, str]:
        """Return whether symbol matches a targeted feedback sector/symbol."""
        symbol = symbol.upper()
        target_symbols = {str(s).upper() for s in policy.get("target_symbols", [])}
        if symbol in target_symbols:
            return True, f"{symbol} was explicitly named in user feedback"

        sector = SECTOR_MAP.get(symbol, "unknown")
        target_sectors = {str(s).lower() for s in policy.get("target_sectors", [])}

        if sector in target_sectors:
            return True, f"{symbol} is classified as sector '{sector}'"

        for target in target_sectors:
            if sector in TARGET_SECTOR_ALIASES.get(target, set()):
                return True, f"{symbol} sector '{sector}' matches target sector '{target}'"

        return False, f"{symbol} sector is '{sector}', not targeted by feedback"

    def apply_feedback_policy_node(self, state: TradingState) -> TradingState:
        """Apply clear user feedback as enforceable policy.

        This node fixes the previous missing concept:
        - LLM/Judge may see feedback but ignore it.
        - This deterministic node enforces direct feedback.
        - It also logs when feedback was considered but not applicable.
        """
        decision = state.get("decision", {"action": "HOLD", "confidence": 0.0})
        policy = state.get("feedback_policy") or (state.get("feedback_context", {}) or {}).get("policy", {}) or {}
        effects = state.setdefault("feedback_effects", [])

        symbol = state.get("symbol", "").upper()
        action = str(decision.get("action", "HOLD")).upper()
        news_signal = state.get("news_signal", {}) or {}
        price_snapshot = state.get("price_snapshot", {}) or {}
        current_position = state.get("current_position", {}) or {}
        owned_quantity = self._owned_quantity(current_position)

        def change_decision(new_action: str, reason: str, confidence_floor: float = 0.71) -> dict[str, Any]:
            effects.append(reason)
            return {
                **decision,
                "original_action_before_feedback": action,
                "action": new_action,
                "confidence": (
                    min(float(decision.get("confidence", 0.0)), 0.70)
                    if new_action == "HOLD"
                    else max(float(decision.get("confidence", 0.0)), confidence_floor)
                ),
                "decision_source": "feedback_policy_override",
                "feedback_policy_applied": True,
                "feedback_policy_reason": reason,
                "rationale": (
                    f"User feedback policy changed the decision to {new_action}. "
                    f"{reason} Original rationale: {decision.get('rationale', '')}"
                ),
            }

        # Always show policy notes as considered feedback.
        for note in policy.get("notes", []) or []:
            if note not in effects:
                effects.append(note)

        if policy.get("force_hold"):
            state["decision"] = change_decision("HOLD", "Feedback requested no trading / HOLD.")
            return state

        # Targeted sell policy: e.g. "Sell all oil and gas positions".
        if policy.get("force_sell_target_if_position"):
            matches_target, match_reason = self._symbol_matches_feedback_targets(symbol, policy)
            if matches_target:
                if owned_quantity > 0 and not policy.get("block_sell"):
                    state["decision"] = change_decision(
                        "SELL",
                        (
                            f"Targeted feedback asks to sell/reduce matching exposure. "
                            f"{match_reason}; owned quantity is {owned_quantity}."
                        ),
                        confidence_floor=0.85,
                    )
                    return state

                effects.append(
                    f"Targeted feedback considered for {symbol}: {match_reason}, "
                    f"but owned quantity is {owned_quantity}; no SELL can be placed."
                )
            else:
                effects.append(
                    f"Targeted feedback considered for {symbol}: {match_reason}; no action applied."
                )

        # Generic sell existing positions when user asks to reduce exposure.
        if policy.get("force_sell_if_position") and not policy.get("force_sell_target_if_position"):
            if owned_quantity > 0 and not policy.get("block_sell"):
                # For generic "sell/reduce", require some weakness unless user explicitly says sell all positions.
                weak_market = price_snapshot.get("trend") == "down" or news_signal.get("signal") in {"negative", "neutral"}
                if weak_market:
                    state["decision"] = change_decision(
                        "SELL",
                        (
                            "Feedback asks to sell/reduce existing positions, an open "
                            f"position exists ({owned_quantity}), and market/news evidence is weak."
                        ),
                        confidence_floor=0.80,
                    )
                    return state
                effects.append(
                    f"Feedback asks to reduce positions, but {symbol} is not currently weak enough to force SELL."
                )
            else:
                effects.append(
                    f"Feedback asks to reduce positions, but {symbol} has no open position to sell."
                )

        # BUY blocking/constraint policies.
        if action == "BUY":
            if policy.get("block_buy"):
                state["decision"] = change_decision("HOLD", "Feedback blocks BUY decisions.")
                return state

            if policy.get("require_positive_news_for_buy") and news_signal.get("signal") != "positive":
                state["decision"] = change_decision("HOLD", "Feedback requires positive news before BUY.")
                return state

            if policy.get("require_uptrend_for_buy") and price_snapshot.get("trend") != "up":
                state["decision"] = change_decision("HOLD", "Feedback requires an uptrend before BUY.")
                return state

            min_news_conf = policy.get("min_news_confidence_for_buy")
            if min_news_conf is not None and float(news_signal.get("confidence", 0.0)) < float(min_news_conf):
                state["decision"] = change_decision(
                    "HOLD",
                    (
                        f"Feedback requires news confidence >= {float(min_news_conf):.2f} before BUY; "
                        f"current news confidence is {float(news_signal.get('confidence', 0.0)):.2f}."
                    ),
                )
                return state

        if action == "SELL" and policy.get("block_sell"):
            state["decision"] = change_decision("HOLD", "Feedback blocks SELL decisions.")
            return state

        # Reduce over-conservative HOLD behavior when all evidence strongly aligns.
        # This is not feedback-driven; it corrects the LLM/Judge tendency to HOLD forever.
        if action == "HOLD" and not policy.get("force_hold"):
            trend = price_snapshot.get("trend")
            signal = news_signal.get("signal")
            news_conf = float(news_signal.get("confidence", 0.0))
            original = state.get("pre_judge_decision", {}) or {}

            # Recover BUY when the LLM originally found a BUY, Judge changed to HOLD,
            # and objective signals are aligned.
            if (
                str(original.get("action", "")).upper() == "BUY"
                and trend == "up"
                and signal == "positive"
                and news_conf >= 0.65
                and not policy.get("block_buy")
            ):
                reason = (
                    "Alignment override: LLM proposed BUY, but Judge changed to HOLD. "
                    "Price trend is up and news signal is positive with sufficient confidence."
                )
                effects.append(reason)
                state["decision"] = {
                    **decision,
                    "original_action_before_alignment_override": "HOLD",
                    "action": "BUY",
                    "confidence": max(float(original.get("confidence", decision.get("confidence", 0.0))), 0.72),
                    "decision_source": "alignment_policy_override",
                    "alignment_policy_applied": True,
                    "rationale": f"{reason} Original Judge rationale: {decision.get('rationale', '')}",
                }
                return state

            # Recover SELL only when the account owns the symbol.
            if (
                str(original.get("action", "")).upper() == "SELL"
                and trend == "down"
                and signal in {"negative", "neutral"}
                and owned_quantity > 0
                and not policy.get("block_sell")
            ):
                reason = (
                    "Alignment override: LLM proposed SELL, but Judge changed to HOLD. "
                    "Price trend is down, news is not positive, and an open position exists."
                )
                effects.append(reason)
                state["decision"] = {
                    **decision,
                    "original_action_before_alignment_override": "HOLD",
                    "action": "SELL",
                    "confidence": max(float(original.get("confidence", decision.get("confidence", 0.0))), 0.72),
                    "decision_source": "alignment_policy_override",
                    "alignment_policy_applied": True,
                    "rationale": f"{reason} Original Judge rationale: {decision.get('rationale', '')}",
                }
                return state

        # If no policy changed the decision, record that feedback was observed.
        if policy.get("notes") and not decision.get("feedback_policy_applied"):
            decision = {
                **decision,
                "feedback_policy_applied": False,
                "feedback_policy_notes": policy.get("notes", []),
            }

        state["decision"] = decision
        return state

    def risk_check_node(self, state: TradingState) -> TradingState:
        decision = state.get("decision", {"action": "HOLD", "confidence": 0.0})
        price_snapshot = state.get("price_snapshot", {})
        account_summary = state.get("account_summary", {})
        current_position = state.get("current_position", {}) or {}

        action = str(decision.get("action", "HOLD")).upper()
        confidence = float(decision.get("confidence", 0.0))

        if account_summary.get("error"):
            risk = {"approved": False, "quantity": 0, "reason": "Account data unavailable."}
        elif account_summary.get("trading_blocked") or account_summary.get("account_blocked"):
            risk = {"approved": False, "quantity": 0, "reason": "Trading/account is blocked by broker."}
        elif action in {"HOLD", "SKIP"}:
            risk = {"approved": False, "quantity": 0, "reason": f"No order required for action {action}."}
        elif action == "SELL" and float(current_position.get("quantity", current_position.get("qty", 0)) or 0) <= 0:
            risk = {
                "approved": False,
                "quantity": 0,
                "reason": "SELL rejected because no current position exists; short selling is disabled.",
            }
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

                if action == "SELL":
                    owned_quantity = int(float(current_position.get("quantity", current_position.get("qty", 0)) or 0))
                    if owned_quantity > 0 and quantity > owned_quantity:
                        quantity = owned_quantity
                        reason += f" SELL quantity capped to owned position ({owned_quantity})."

                feedback_policy = (state.get("feedback_context", {}) or {}).get("policy", {}) or {}
                max_feedback_qty = feedback_policy.get("max_order_quantity")
                if max_feedback_qty is not None and quantity > int(max_feedback_qty):
                    quantity = int(max_feedback_qty)
                    reason += f" User feedback capped order quantity at {quantity}."
                    state.setdefault("feedback_effects", []).append(
                        f"Feedback capped order quantity at {quantity}."
                    )

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
                "feedback_context": state.get("feedback_context", {}),
                "feedback_effects": state.get("feedback_effects", []),
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
        builder.add_node("judge_decision", self.judge_decision_node)
        builder.add_node("read_feedback", self.read_feedback_node)
        builder.add_node("apply_feedback_policy", self.apply_feedback_policy_node)

        builder.set_entry_point("get_account")
        builder.add_edge("get_account", "get_market_data")
        builder.add_edge("get_market_data", "get_news")
        builder.add_edge("get_news", "read_feedback")
        builder.add_edge("read_feedback", "llm_reasoning")
        builder.add_edge("llm_reasoning", "judge_decision")
        builder.add_edge("judge_decision", "apply_feedback_policy")
        builder.add_edge("apply_feedback_policy", "risk_check")
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
