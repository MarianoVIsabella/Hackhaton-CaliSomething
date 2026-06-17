

from __future__ import annotations

from typing import Any

from src.graph.trading_graph import TradingGraph


class TradingAgent:
    def __init__(
        self,
        symbols: list[str],
        execute_orders: bool = False,
        max_position_value_pct: float = 0.10,
        confidence_threshold: float = 0.70,
        demo_trade_mode: bool = False,
    ) -> None:
        self.symbols = [symbol.upper().strip() for symbol in symbols]
        self.graph = TradingGraph(
            execute_orders=execute_orders,
            demo_trade_mode=demo_trade_mode,
            max_position_value_pct=max_position_value_pct,
            confidence_threshold=confidence_threshold,
        )
        self.journal = self.graph.journal

    def run_cycle_for_symbol(self, symbol: str) -> dict[str, Any]:
        final_state = self.graph.run_symbol(symbol)
        return final_state.get("journal_entry", final_state)

    def run_once(self) -> list[dict[str, Any]]:
        return [self.run_cycle_for_symbol(symbol) for symbol in self.symbols]


if __name__ == "__main__":
    agent = TradingAgent(symbols=["AAPL", "MSFT", "NVDA", "TSLA"], execute_orders=False)
    results = agent.run_once()
    for result in results:
        print(result)
    print("\nJournal summary:")
    print(agent.journal.summarize())
