
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


class TradeJournal:
    """
    JSONL-based trade journal.

    Each decision is stored as one JSON object per line.
    This format is simple, readable, append-only, and hackathon-friendly.
    """

    def __init__(self, journal_path: str = "logs/trade_journal.jsonl") -> None:
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.touch(exist_ok=True)

    def log_decision(
        self,
        symbol: str,
        action: str,
        rationale: str,
        confidence: float,
        price_data: dict[str, Any] | None = None,
        news_data: dict[str, Any] | None = None,
        risk_check: dict[str, Any] | None = None,
        order_result: dict[str, Any] | None = None,
        portfolio_state: dict[str, Any] | None = None,
        error_notes: str | None = None,
    ) -> dict[str, Any]:
        """
        Record one complete agent decision.

        action should usually be:
        - BUY
        - SELL
        - HOLD
        - SKIP
        """

        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol.upper().strip(),
            "action": action.upper().strip(),
            "confidence": round(float(confidence), 4),
            "rationale": rationale,
            "price_data": price_data or {},
            "news_data": news_data or {},
            "risk_check": risk_check or {},
            "order_result": order_result or {},
            "portfolio_state": portfolio_state or {},
            "error_notes": error_notes,
        }

        with self.journal_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def read_all(self) -> list[dict[str, Any]]:
        """
        Read all journal entries.
        """
        entries: list[dict[str, Any]] = []

        with self.journal_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        return entries

    def read_last(self, n: int = 5) -> list[dict[str, Any]]:
        """
        Read the last n journal entries.
        """
        entries = self.read_all()
        return entries[-n:]

    def summarize(self) -> dict[str, Any]:
        """
        Produce a compact summary of the journal.
        """
        entries = self.read_all()

        total = len(entries)
        buys = sum(1 for e in entries if e.get("action") == "BUY")
        sells = sum(1 for e in entries if e.get("action") == "SELL")
        holds = sum(1 for e in entries if e.get("action") == "HOLD")
        skips = sum(1 for e in entries if e.get("action") == "SKIP")
        errors = sum(1 for e in entries if e.get("error_notes"))

        symbols = sorted(set(e.get("symbol") for e in entries if e.get("symbol")))

        return {
            "journal_path": str(self.journal_path),
            "total_entries": total,
            "buy_decisions": buys,
            "sell_decisions": sells,
            "hold_decisions": holds,
            "skip_decisions": skips,
            "entries_with_errors": errors,
            "symbols_seen": symbols,
        }


if __name__ == "__main__":
    journal = TradeJournal()

    test_entry = journal.log_decision(
        symbol="AAPL",
        action="HOLD",
        confidence=0.65,
        rationale=(
            "Test journal entry. The agent did not trade because this is "
            "a connectivity and logging test."
        ),
        price_data={
            "latest_price": 291.08,
            "source": "test_data",
        },
        risk_check={
            "approved": False,
            "reason": "Testing only.",
        },
        order_result={
            "executed": False,
        },
    )

    print("Logged entry:")
    print(test_entry)

    print("\nJournal summary:")
    print(journal.summarize())