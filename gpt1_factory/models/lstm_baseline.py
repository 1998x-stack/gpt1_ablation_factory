from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS


@MODELS.register("lstm_baseline")
class LSTMBaseline(nn.Module):
    """单层 LSTM 作为对照（论文消融用），hidden=2048 默认。"""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        lstm_hidden: int = 2048,
        num_layers: int = 1,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        x = self.tok_emb(input_ids)
        x, _ = self.lstm(x)
        x = self.drop(self.proj(x))
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "last_hidden_state": x, "loss": loss}
