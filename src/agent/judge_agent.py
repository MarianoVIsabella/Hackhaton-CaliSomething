

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()



## Fix: make JudgeAgent robust to messy JSON

def extract_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from an LLM response, even if it is wrapped in markdown.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in response: {text}")

    return json.loads(text[start:end + 1])

class JudgeAgent:
    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "deepseek-v4-pro").strip()

        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"

        elif self.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"

        elif self.provider == "ollama":
            api_key = "ollama"
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

        elif self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if not api_key:
            raise ValueError(f"Missing API key for LLM provider: {self.provider}")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def review(
        self,
        symbol: str,
        price_snapshot: dict[str, Any],
        news_signal: dict[str, Any],
        account_summary: dict[str, Any],
        current_position: dict[str, Any],
        proposed_decision: dict[str, Any],
        feedback_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Review the proposed trading decision.

        Returns a final decision object.
        """

        prompt = {
            "instruction": (
                "You are a strict trading judge agent in an educational paper-trading system. "
                "Your job is to review the proposed trading decision before risk checking. "
                "Use only the provided tool data. Never invent prices, news, positions, or account data. "
                "You may ACCEPT the proposed decision or CHANGE it. "
                "If evidence is weak, contradictory, or unsafe, change the decision to HOLD. "
                "If the proposed action is SELL but there is no current position, change to HOLD unless shorting is explicitly allowed. "
                "If trend is down and the agent owns the symbol, SELL may be appropriate. "
                "Return only valid JSON."
                # "Sell all the positions related or impacted by a crisis in oil and gas market"
            ),
            "symbol": symbol,
            "price_snapshot": price_snapshot,
            "news_signal": news_signal,
            "account_summary": account_summary,
            "current_position": current_position,
            "proposed_decision": proposed_decision,
            "feedback_context": feedback_context or {},
            "feedback_review_rules": [
                "Short-term missions from feedback_context must be enforced on the current decision.",
                "If feedback_context.policy.force_hold is true, change the action to HOLD.",
                "If feedback_context.policy.block_buy is true, reject BUY and change to HOLD.",
                "If feedback_context.policy.block_sell is true, reject SELL and change to HOLD.",
                "If feedback_context.policy.min_news_confidence_for_buy is set, reject BUY when news confidence is below that threshold.",
                "If feedback_context.policy.force_sell_if_position is true and current_position.quantity > 0, SELL may be appropriate unless contradicted by stronger safety rules.",
            ],
            "review_rules": [
                "Change to BUY only if trend is up, news is positive/neutral, and confidence is reasonably high.",
                "Change to SELL only if trend is down and either news is negative or an existing position should be reduced.",
                # "Change to HOLD if signals conflict.",
                "Change to SKIP if required data is missing.",
                "Change to SELL if trend is down and either news is negative or the account already holds the symbol, even if news is neutral.",
            ],
            "output_schema": {
                "action": "BUY | SELL | HOLD | SKIP",
                "confidence": "float from 0.0 to 1.0",
                "rationale": "final rationale grounded in provided data",
                "judge_verdict": "ACCEPTED | CHANGED",
                "judge_reason": "why the judge accepted or changed the decision",
                "original_action": "original proposed action",
            },
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative trading safety judge. "
                            "This is paper trading only, not financial advice. "
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt, indent=2),
                    },
                ],
            )

            content = response.choices[0].message.content or ""
            reviewed = extract_json_object(content)

            action = str(reviewed.get("action", "HOLD")).upper()
            if action not in {"BUY", "SELL", "HOLD", "SKIP"}:
                action = "HOLD"

            confidence = float(reviewed.get("confidence", 0.0))
            confidence = max(0.0, min(confidence, 1.0))

            return {
                "action": action,
                "confidence": confidence,
                "rationale": str(reviewed.get("rationale", "")),
                "decision_source": "judge_agent",
                "judge_verdict": str(reviewed.get("judge_verdict", "CHANGED")),
                "judge_reason": str(reviewed.get("judge_reason", "")),
                "original_action": str(
                    reviewed.get(
                        "original_action",
                        proposed_decision.get("action", "UNKNOWN"),
                    )
                ),
                "proposed_decision": proposed_decision,
            }

        except Exception as error:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "rationale": (
                    f"Judge Agent failed: {error}. Defaulting to HOLD for safety."
                ),
                "decision_source": "judge_failure_safe_hold",
                "judge_verdict": "CHANGED",
                "judge_reason": "Judge failure; safe fallback to HOLD.",
                "original_action": proposed_decision.get("action", "UNKNOWN"),
                "proposed_decision": proposed_decision,
            }