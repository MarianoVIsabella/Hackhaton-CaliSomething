"""

The agent runs independently for a fixed duration and repeatedly executes this
explicit graph:
get_account -> get_market_data -> get_news -> read_feedback -> llm_reasoning -> judge_decision -> apply_feedback_policy -> risk_check -> execute_order -> write_journal
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.graph.trading_graph import TradingGraph


load_dotenv()


def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def run_autonomous_demo(
    duration_minutes: int = 5,
    cycle_sleep_seconds: int = 60,
) -> None:
    symbols = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,NVDA,TSLA").split(",") if s.strip()]
    execute_orders = env_bool("PAPER_TRADING_ENABLED", "false")
    demo_trade_mode = env_bool("DEMO_TRADE_MODE", "false")

    trading_graph = TradingGraph(
        execute_orders=execute_orders,
        demo_trade_mode=demo_trade_mode,
        max_position_value_pct=float(os.getenv("MAX_POSITION_VALUE_PCT", "0.10")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
        demo_quantity=int(os.getenv("DEMO_QUANTITY", "1")),
    )

    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    cycle_number = 1

    print("\n====================================")
    print(" LANGGRAPH AUTONOMOUS TRADING AGENT")
    print("====================================")
    print(f"Start time UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Symbols: {symbols}")
    print(f"Mode: {'PAPER TRADING' if execute_orders else 'DRY RUN'}")
    print(f"Demo trade mode: {demo_trade_mode}")
    print("Graph flow:")
    print("get_account -> get_market_data -> get_news -> read_feedback -> llm_reasoning -> judge_decision -> apply_feedback_policy -> risk_check -> execute_order -> write_journal")
    print("====================================\n")

    while time.time() < end_time:
        print(f"\n========== Graph Cycle {cycle_number} ==========")

        for symbol in symbols:
            print(f"\n--- Running LangGraph cycle for {symbol} ---")
            try:
                final_state = trading_graph.run_symbol(symbol)

                decision = final_state.get("decision", {})
                price_snapshot = final_state.get("price_snapshot", {})
                news_signal = final_state.get("news_signal", {})
                risk = final_state.get("risk_check", {})
                order = final_state.get("order_result", {})
                feedback_context = final_state.get("feedback_context", {})
                feedback_effects = final_state.get("feedback_effects", [])

                if execute_orders:
                    print("⚠ PAPER TRADING ENABLED (orders may be sent to Alpaca Paper Account)")

                print("Price trend:", price_snapshot.get("trend"))
                print("Latest price:", price_snapshot.get("latest_price"))
                print("Bars source:", price_snapshot.get("bars_source"))
                print("Decision:", decision.get("action"))
                print("Decision source:", decision.get("decision_source"))
                print("Confidence:", decision.get("confidence"))
                print("News signal:", news_signal)
                print("Feedback missions:", {
                    "short_term": feedback_context.get("short_term_missions", []),
                    "long_term": feedback_context.get("long_term_missions", []),
                    "general": feedback_context.get("general_feedback", []),
                })
                print("Feedback effects:", feedback_effects)
                print("Rationale:", decision.get("rationale"))
                print("Risk:", risk.get("reason"))
                print("Order:", order)

            except Exception as error:
                print(f"Graph cycle failed for {symbol}, but runner recovered: {error}")

        print("\nJournal summary:")
        print(trading_graph.journal.summarize())

        cycle_number += 1
        remaining = end_time - time.time()
        if remaining <= 0:
            break

        sleep_time = min(cycle_sleep_seconds, remaining)
        print(f"\nSleeping for {int(sleep_time)} seconds before next graph cycle...")
        time.sleep(sleep_time)

    print("\n====================================")
    print(" LANGGRAPH TRADING AGENT FINISHED")
    print("====================================")
    print(f"End time UTC: {datetime.now(timezone.utc).isoformat()}")
    print("Final journal summary:")
    print(trading_graph.journal.summarize())


if __name__ == "__main__":
    run_autonomous_demo(
        duration_minutes=int(os.getenv("DEMO_DURATION_MINUTES", "5")),
        cycle_sleep_seconds=int(os.getenv("CYCLE_SLEEP_SECONDS", "60")),
    )
