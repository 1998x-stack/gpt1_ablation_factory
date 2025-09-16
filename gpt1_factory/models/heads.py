from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """通用分类头，给 LSTM/GPT 的 last_hidden_state 使用。"""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = last_hidden_state[:, -1, :]
        x = self.drop(x)
        return self.fc(x)
