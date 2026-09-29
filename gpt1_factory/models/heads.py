from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def pool_sequence_state(
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    classify_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select the task representation from an encoded sequence.

    Explicit classify positions take priority. Otherwise the last non-padding
    token is used, which avoids silently pooling a right-padding position.
    """

    batch_size, seq_len, _ = last_hidden_state.shape

    if classify_positions is not None:
        if classify_positions.ndim != 1 or classify_positions.size(0) != batch_size:
            raise ValueError("classify_positions must have shape [batch].")
        if torch.any(classify_positions < 0) or torch.any(classify_positions >= seq_len):
            raise ValueError("classify_positions contains an out-of-range index.")
        positions = classify_positions.to(last_hidden_state.device)
    elif attention_mask is not None:
        if attention_mask.shape != last_hidden_state.shape[:2]:
            raise ValueError("attention_mask must have shape [batch, sequence].")
        indices = torch.arange(seq_len, device=last_hidden_state.device).unsqueeze(0)
        positions = (
            indices.expand(batch_size, seq_len)
            .masked_fill(attention_mask.to(last_hidden_state.device) == 0, -1)
            .max(dim=-1)
            .values
            .clamp_min(0)
        )
    else:
        positions = torch.full(
            (batch_size,),
            seq_len - 1,
            dtype=torch.long,
            device=last_hidden_state.device,
        )

    batch = torch.arange(batch_size, device=last_hidden_state.device)
    return last_hidden_state[batch, positions]


class ClassificationHead(nn.Module):
    """Linear classifier over an explicit or last-valid sequence state."""

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        classify_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = pool_sequence_state(
            last_hidden_state,
            attention_mask=attention_mask,
            classify_positions=classify_positions,
        )
        return self.fc(self.drop(x))


class ChoiceScoringHead(nn.Module):
    """Shared scalar scorer for multiple-choice candidate representations."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, choice_hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(choice_hidden_state)).squeeze(-1)
