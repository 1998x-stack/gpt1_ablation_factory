from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


class ZeroShotHeuristics:
    """论文里的零样本启发式评估（简化版）。
    - SST2: 比较 "positive"/"negative" 的续写概率
    - CoLA: 平均 token 对数概率阈值（此处仅示例，实际需要语法错误集）
    """

    def __init__(self, backbone: torch.nn.Module, tokenizer) -> None:
        self.m = backbone
        self.tok = tokenizer
        self.device = next(self.m.parameters()).device

    @torch.no_grad()
    def sst2(self, sentence: str) -> int:
        prompt_pos = f"{sentence} It was positive ."
        prompt_neg = f"{sentence} It was negative ."
        prob_pos = self._avg_log_prob(prompt_pos)
        prob_neg = self._avg_log_prob(prompt_neg)
        return 1 if prob_pos > prob_neg else 0

    def _avg_log_prob(self, text: str) -> float:
        ids = self.tok.encode(text).ids
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.m(input_ids=x, labels=y)
        return -float(out["loss"])
