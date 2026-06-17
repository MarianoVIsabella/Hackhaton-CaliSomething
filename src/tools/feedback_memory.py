"""
feedback_memory.py

Persistent feedback memory for the trading graph.

The FeedbackAgent writes classified user feedback to logs/user_feedback.jsonl.
This module reads that feedback and converts it into a compact context object
that the Trader/Judge agents and deterministic policy node can use.

Important design:
- Feedback collection alone is not enough.
- Feedback must be converted into an enforceable policy.
- The graph must log whether feedback was applied, ignored, or not applicable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class FeedbackMemory:
    def __init__(self, path: str = "logs/user_feedback.jsonl") -> None:
        self.path = Path(path)

    def read_latest_summary(self, limit: int = 10) -> dict[str, Any]:
        """
        Return the latest user feedback grouped by mission type plus policy.

        Expected shape:
        {
            "short_term_missions": [...],
            "long_term_missions": [...],
            "general_feedback": [...],
            "policy": {...}
        }
        """
        result: dict[str, Any] = {
            "short_term_missions": [],
            "long_term_missions": [],
            "general_feedback": [],
            "policy": self._empty_policy(),
        }

        if not self.path.exists():
            return result

        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]

        for line in lines:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            classified = entry.get("classified_feedback", {})

            # Support both old and new key naming just in case.
            key_map = {
                "short_term_missions": ["short_term_missions", "short_term"],
                "long_term_missions": ["long_term_missions", "long_term"],
                "general_feedback": ["general_feedback", "general"],
            }

            for canonical_key, possible_keys in key_map.items():
                for key in possible_keys:
                    items = classified.get(key, [])
                    if isinstance(items, list):
                        result[canonical_key].extend(
                            str(item).strip() for item in items if str(item).strip()
                        )

        result["policy"] = self.extract_action_policy(result)
        return result

    @staticmethod
    def _empty_policy() -> dict[str, Any]:
        return {
            "block_buy": False,
            "block_sell": False,
            "force_hold": False,

            # Generic sell/reduce existing position.
            "force_sell_if_position": False,

            # Targeted sell/reduce policy.
            "force_sell_target_if_position": False,
            "target_symbols": [],
            "target_sectors": [],

            # BUY constraints.
            "require_positive_news_for_buy": False,
            "require_uptrend_for_buy": False,
            "min_news_confidence_for_buy": None,

            # Risk/quantity constraints.
            "max_order_quantity": None,

            # Human-readable policy notes.
            "notes": [],
        }

    def extract_action_policy(self, feedback_context: dict[str, Any]) -> dict[str, Any]:
        """
        Convert natural-language feedback into simple enforceable rules.

        Examples it understands:
        - "Do not buy if news confidence is below 0.7"
        - "Sell all oil and gas positions"
        - "Exit energy stocks because of an oil crisis"
        - "Use smaller quantity"
        - "Do not trade this cycle"
        """
        policy = self._empty_policy()

        all_items: list[str] = []
        for key in ["short_term_missions", "long_term_missions", "general_feedback"]:
            all_items.extend(feedback_context.get(key, []) or [])

        text = "\n".join(all_items).lower()
        if not text.strip():
            return policy

        # Global no-trade / hold policies.
        if any(phrase in text for phrase in [
            "do not trade", "don't trade", "avoid trading", "no trading",
            "force hold", "stay out of the market"
        ]):
            policy["force_hold"] = True
            policy["notes"].append("User feedback asks the agent to avoid trading / HOLD.")

        # Global action blocks.
        if any(phrase in text for phrase in [
            "do not buy", "don't buy", "avoid buying", "no buy", "stop buying"
        ]):
            policy["block_buy"] = True
            policy["notes"].append("User feedback blocks BUY decisions.")

        if any(phrase in text for phrase in [
            "do not sell", "don't sell", "avoid selling", "no sell", "stop selling"
        ]):
            policy["block_sell"] = True
            policy["notes"].append("User feedback blocks SELL decisions.")

        # Generic sell existing position.
        if any(phrase in text for phrase in [
            "sell if position", "sell existing", "reduce position", "exit position",
            "sell the position", "sell all positions", "liquidate positions",
            "close positions"
        ]):
            policy["force_sell_if_position"] = True
            policy["notes"].append("User feedback asks to sell/reduce existing positions when applicable.")

        # Target sectors: oil/gas/energy crisis.
        oil_gas_terms = [
            "oil", "gas", "oil and gas", "oil/gas", "energy crisis",
            "energy market", "oil market", "gas market", "crude", "petroleum"
        ]
        if any(term in text for term in oil_gas_terms):
            policy["target_sectors"].append("oil_gas")
            policy["notes"].append("User feedback refers to oil/gas/energy exposure.")

            if any(word in text for word in ["sell", "exit", "reduce", "liquidate", "close"]):
                policy["force_sell_target_if_position"] = True
                policy["notes"].append("User feedback asks to sell/reduce oil/gas-related positions.")

        # Target symbols if explicitly mentioned.
        # This catches uppercase stock tickers in raw feedback better than lower text,
        # so rebuild from original.
        raw_text = "\n".join(all_items)
        symbol_candidates = re.findall(r"\b[A-Z]{1,5}\b", raw_text)
        ignore_words = {"BUY", "SELL", "HOLD", "SKIP", "ETF", "AI", "API", "LLM"}
        for sym in symbol_candidates:
            if sym not in ignore_words and sym not in policy["target_symbols"]:
                policy["target_symbols"].append(sym)

        if policy["target_symbols"] and any(word in text for word in ["sell", "exit", "reduce", "liquidate", "close"]):
            policy["force_sell_target_if_position"] = True
            policy["notes"].append(
                f"User feedback targets symbols: {', '.join(policy['target_symbols'])}."
            )

        # BUY constraints.
        if "positive news" in text and any(phrase in text for phrase in ["only buy", "buy only", "require"]):
            policy["require_positive_news_for_buy"] = True
            policy["notes"].append("User feedback requires positive news before buying.")

        if ("uptrend" in text or "up trend" in text or "trend is up" in text) and any(
            phrase in text for phrase in ["only buy", "buy only", "require"]
        ):
            policy["require_uptrend_for_buy"] = True
            policy["notes"].append("User feedback requires an uptrend before buying.")

        # Extract thresholds from feedback such as:
        # "news confidence below 0.7", "below 70%", "confidence < 0.75".
        threshold = None
        decimal_match = re.search(
            r"news confidence[^\n]*?(?:below|under|less than|<)\s*(0\.\d+|1\.0|1)",
            text,
        )
        percent_match = re.search(
            r"news confidence[^\n]*?(?:below|under|less than|<)\s*(\d{1,3})\s*%",
            text,
        )
        generic_decimal = re.search(
            r"(?:confidence|news)[^\n]*?(?:below|under|less than|<)\s*(0\.\d+|1\.0|1)",
            text,
        )

        if decimal_match:
            threshold = float(decimal_match.group(1))
        elif percent_match:
            threshold = float(percent_match.group(1)) / 100.0
        elif generic_decimal:
            threshold = float(generic_decimal.group(1))

        if threshold is not None:
            threshold = max(0.0, min(float(threshold), 1.0))
            policy["min_news_confidence_for_buy"] = threshold
            policy["notes"].append(
                f"User feedback requires news confidence >= {threshold:.2f} for BUY decisions."
            )

        # Quantity cap.
        quantity_match = re.search(
            r"(?:max|maximum|limit|only)\s*(\d+)\s*(?:share|shares|unit|units)",
            text,
        )
        if quantity_match:
            policy["max_order_quantity"] = max(1, int(quantity_match.group(1)))
            policy["notes"].append(
                f"User feedback limits order quantity to {policy['max_order_quantity']} share(s)."
            )
        elif any(phrase in text for phrase in [
            "smaller quantity", "buy less", "sell less", "tiny order", "small order"
        ]):
            policy["max_order_quantity"] = 1
            policy["notes"].append("User feedback asks for smaller/tiny orders; quantity capped at 1.")

        return policy
