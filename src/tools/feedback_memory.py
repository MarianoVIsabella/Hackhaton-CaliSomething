import json
from pathlib import Path


class FeedbackMemory:
    def __init__(self, path: str = "logs/user_feedback.jsonl") -> None:
        self.path = Path(path)

    def read_latest_summary(self, limit: int = 10) -> dict:
        result = {
            "short_term_missions": [],
            "long_term_missions": [],
            "general_feedback": [],
        }

        if not self.path.exists():
            return result

        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]

        for line in lines:
            if not line.strip():
                continue

            entry = json.loads(line)
            classified = entry.get("classified_feedback", {})

            for key in result:
                result[key].extend(classified.get(key, []))

        return result