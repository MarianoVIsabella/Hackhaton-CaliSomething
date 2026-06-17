
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMReasoningAgent:
    """Provider-neutral OpenAI-compatible LLM reasoning wrapper.

    Supported providers:
    - OpenAI: LLM_PROVIDER=openai, OPENAI_API_KEY=..., LLM_MODEL=gpt-4o-mini
    - Groq:   LLM_PROVIDER=groq,   GROQ_API_KEY=...,   LLM_MODEL=llama-3.1-8b-instant

    Groq is OpenAI-compatible through its OpenAI-compatible base URL.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
        self.model = os.getenv("MODEL_NAME", "deepseek-v4-pro").strip()

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
            raise ValueError(
                f"Missing API key for LLM provider '{self.provider}'. "
                "Set OPENAI_API_KEY or GROQ_API_KEY in .env."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Parse JSON even if the model wraps it in markdown fences."""
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def decide(
        self,
        symbol: str,
        price_snapshot: dict[str, Any],
        news_signal: dict[str, Any],
        account_summary: dict[str, Any],
        current_position: dict[str, Any],
        feedback_context: dict[str, list[str]],
        all_positions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a BUY/SELL/HOLD/SKIP decision as a validated dictionary."""
        prompt = {
            "mission": (
                "You are an expert trading agent analyst who reasons marketing decisions based on price data, market news, user feedbacks and account summary. "
                "Your reasoning should also consider what would big investors do."
                "Decide one action: "
                "BUY, SELL, HOLD, or SKIP."
            ),
            "strict_rules": [
                "Use only tool-provided data below.",
                "Never invent prices, news, account values, or positions.",
                "If price data is missing, contradictory, or weak, choose HOLD or SKIP.",
                "BUY when price trend is up AND news signal is positive.",
                "SELL when price trend is down AND news signal is neutral/negative.",
                "SELL when price trend is down AND the account already holds this symbol, even if news is neutral.",
                "HOLD when confidence is too low.",
                "SKIP only when required tool data is missing or invalid.",
            ],
            "symbol": symbol,
            "price_snapshot": price_snapshot,
            "news_signal": news_signal,
            "account_summary": account_summary,
            "current_position": current_position,
            "feedback_context": feedback_context,
            "all_positions": all_positions or [],
            "feedback_instructions": [
                "You must explicitly consider feedback_context before deciding.",
                "Short-term missions directly constrain the current/next decision.",
                "Long-term missions influence strategy and risk appetite over time.",
                "General feedback informs explanation quality and user preferences.",
                "If feedback_context.policy.force_hold is true, choose HOLD.",
                "If feedback_context.policy.block_buy is true, do not choose BUY.",
                "If feedback_context.policy.block_sell is true, do not choose SELL.",
                "If feedback_context.policy.min_news_confidence_for_buy is set, only BUY when news confidence is at least that threshold.",
            ],
            "required_json_output": {
                "action": "BUY | SELL | HOLD | SKIP",
                "confidence": "float from 0.0 to 1.0",
                "rationale": "brief explanation grounded in the provided data",
            },
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_completion_tokens=300,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious autonomous trading reasoning node. "
                            "Return only valid JSON. Do not include markdown."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
            )

            content = response.choices[0].message.content or "{}"
            raw_decision = self._extract_json(content)

            action = str(raw_decision.get("action", "HOLD")).upper().strip()
            confidence = float(raw_decision.get("confidence", 0.0))
            rationale = str(raw_decision.get("rationale", "")).strip()

            if action not in {"BUY", "SELL", "HOLD", "SKIP"}:
                action = "HOLD"
            confidence = max(0.0, min(confidence, 1.0))
            if not rationale:
                rationale = "LLM returned no rationale; defaulting to HOLD."

            return {
                "action": action,
                "confidence": confidence,
                "rationale": rationale,
                "decision_source": "llm_reasoning_node",
                "provider": self.provider,
                "model": self.model,
            }

        except Exception as error:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "rationale": (
                    f"LLM reasoning failed: {error}. The graph defaults to HOLD "
                    "to avoid unsupported trading."
                ),
                "decision_source": "llm_failure_safe_hold",
                "provider": self.provider,
                "model": self.model,
            }
