from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TaskFormatter:
    """Serialize structured inputs into sequences per the paper for the shared LM.

    - NLI: "premise <sep> hypothesis"
    - Similarity/paraphrase: run both orders once and add (fields built before the collator; extensible for finer control)
    - QA: [doc; question; <sep>; option_k] once per option
    """
    mode: str = "classification"  # or "qa", "similarity"

    def format_pair(self, a: str, b: str | None) -> str:
        if b is None:
            return f"<s> {a}"
        return f"<s> {a} <sep> {b}"
