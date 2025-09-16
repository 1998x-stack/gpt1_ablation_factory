from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class TaskFormatter:
    """将结构化输入按论文策略顺序化拼接，交给统一 LM 处理。

    - NLI: "premise <sep> hypothesis"
    - 相似度/复述: 双序过一遍再相加（此处在 collator 前构造字段；如需更细控制可扩展）
    - QA:  [doc; question; <sep>; option_k] 对每个选项各一次
    """
    mode: str = "classification"  # or "qa", "similarity"

    def format_pair(self, a: str, b: str | None) -> str:
        if b is None:
            return f"<s> {a}"
        return f"<s> {a} <sep> {b}"
