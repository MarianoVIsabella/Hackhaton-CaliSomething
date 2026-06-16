"""
feedback_agent.py

User Feedback Agent.

This agent waits for user feedback, classifies it into:
1. short_term_missions
2. long_term_missions
3. general_feedback

Then stores the result as structured JSON.
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
        """
        Rule-based feedback classifier.

        This is intentionally lightweight and reliable for hackathon use.
        Later it can be upgraded with an LLM classifier.
        """

        feedback = feedback.strip()

        result = {
            "short_term_missions": [],
            "long_term_missions": [],
            "general_feedback": [],
        }

        if not feedback:
            return result

        lower_feedback = feedback.lower()

        short_term_keywords = [
            "next decision",
            "next trade",
            "immediately",
            "now",
            "today",
            "this cycle",
            "increase confidence",
            "reduce confidence",
            "buy less",
            "sell",
            "hold",
            "avoid",
            "use smaller quantity",
            "check price",
            "check news",
            "do not trade",
        ]

        long_term_keywords = [
            "over time",
            "long term",
            "strategy",
            "recurring",
            "learn",
            "adapt",
            "risk management",
            "diversification",
            "portfolio",
            "trend",
            "memory",
            "improve future",
            "historical",
            "performance",
        ]

        if any(keyword in lower_feedback for keyword in short_term_keywords):
            result["short_term_missions"].append(feedback)

        elif any(keyword in lower_feedback for keyword in long_term_keywords):
            result["long_term_missions"].append(feedback)

        else:
            result["general_feedback"].append(feedback)

        return result

    def save_feedback(
        self,
        raw_feedback: str,
        classified_feedback: dict[str, list[str]],
    ) -> dict[str, Any]:
        """
        Save feedback classification to JSONL.
        """

        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "raw_feedback": raw_feedback,
            "classified_feedback": classified_feedback,
        }

        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def process_feedback(self, feedback: str) -> dict[str, Any]:
        """
        Classify and save one feedback item.
        """

        classified = self.classify_feedback(feedback)
        return self.save_feedback(feedback, classified)

    def run_interactive(self) -> None:
        """
        Wait for user input continuously.
        Type 'exit' to stop.
        """

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