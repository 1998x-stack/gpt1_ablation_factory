from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Generic classification head used with the LSTM/GPT last-hidden-state output."""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = last_hidden_state[:, -1, :]
        x = self.drop(x)
        return self.fc(x)
