"""
feedback_agent.py

User Feedback Agent.

This agent waits for user feedback, classifies it into:
1. short_term_missions
2. long_term_missions
3. general_feedback

Then stores the result as structured JSONL so the trading graph can read it.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


class FeedbackAgent:
    def __init__(self, output_path: str = "logs/user_feedback.jsonl") -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.touch(exist_ok=True)

    def classify_feedback(self, feedback: str) -> dict[str, list[str]]:
       
        feedback = feedback.strip()

        result = {
            "short_term_missions": [],
            "long_term_missions": [],
            "general_feedback": [],
        }

        if not feedback:
            return result

        lower = feedback.lower()

        # Immediate trading instructions.
        short_term_keywords = [
            "next decision", "next trade", "immediately", "now", "today",
            "this cycle", "buy", "sell", "hold", "skip", "avoid", "exit",
            "reduce", "close", "liquidate", "do not trade", "don't trade",
            "do not buy", "don't buy", "do not sell", "don't sell",
            "use smaller quantity", "max", "maximum", "limit", "only",
            "check price", "check news", "confidence below", "news confidence",
            "oil", "gas", "energy crisis", "crisis", "market shock",  "portfolio",
            "position", "positions",
        ]

        # Longer-term strategy or behavior change.
        long_term_keywords = [
            "over time", "long term", "strategy", "recurring", "learn",
            "adapt", "risk management", "diversification",
            "memory", "improve future", "historical", "performance",
            "in the future", "going forward", "overall", "always prefer",
            "usually", "trend over days", "weekly",
        ]

        if any(k in lower for k in short_term_keywords):
            result["short_term_missions"].append(feedback)
        elif any(k in lower for k in long_term_keywords):
            result["long_term_missions"].append(feedback)
        else:
            result["general_feedback"].append(feedback)

        return result

    def save_feedback(
        self,
        raw_feedback: str,
        classified_feedback: dict[str, list[str]],
    ) -> dict[str, Any]:
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "raw_feedback": raw_feedback,
            "classified_feedback": classified_feedback,
        }

        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def process_feedback(self, feedback: str) -> dict[str, Any]:
        classified = self.classify_feedback(feedback)
        return self.save_feedback(feedback, classified)

    def run_interactive(self) -> None:
        print("\n=== Feedback Agent Started ===")
        print("Type feedback about the trading agent.")
        print("Type 'exit' to stop.\n")

        while True:
            feedback = input("Feedback> ").strip()

            if feedback.lower() in {"exit", "quit", "stop"}:
                print("Feedback Agent stopped.")
                break

            entry = self.process_feedback(feedback)

            print("\nClassified feedback:")
            print(json.dumps(entry["classified_feedback"], indent=2))
            print()


if __name__ == "__main__":
    FeedbackAgent().run_interactive()
