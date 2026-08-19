from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..registry import MODELS


class GELU(nn.Module):
    """Module wrapper around the GELU activation."""
    def forward(self, x):
        """Apply GELU activation.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of the same shape as the input with GELU applied.
        """
        return F.gelu(x)


class CausalSelfAttention(nn.Module):
    """Causal masked self-attention."""

    def __init__(self, d_model: int, n_head: int, attn_dropout: float, resid_dropout: float, max_len: int) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        # causal mask
        mask = torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply causal-masked self-attention.

        Args:
            x: Input sequence representation of shape (B, T, C).
            attn_mask: Optional attention mask of shape (B, T); 0 means masked.

        Returns:
            Updated representation of shape (B, T, C).
        """
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b t (h d) -> b h t d", h=self.n_head) for t in qkv]
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal = self.mask[:, :, :T, :T]
        att = att.masked_fill(causal == 0, float("-inf"))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask[:, None, None, :T] == 0, float("-inf"))

        att = att.softmax(dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer decoder block: residual stack of LN + self-attention + MLP."""
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float, attn_dropout: float, resid_dropout: float, max_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, attn_dropout, resid_dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a single decoder block.

        Args:
            x: (B, T, C) current layer input.
            attn_mask: (B, T) optional attention mask.

        Returns:
            (B, T, C) residual-updated output.
        """
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


@MODELS.register("gpt_decoder")
class GPTDecoderLM(nn.Module):
    """Only-decoder Transformer language model, also usable as a backbone for downstream classification."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_layer: int = 12,
        n_head: int = 12,
        d_model: int = 768,
        d_ff: int = 3072,
        max_len: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        tie_emb: bool = False,
        layer_norm_eps: float = 1e-5,
        gelu: bool = True,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout, attn_dropout, resid_dropout, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_emb:
            self.lm_head.weight = self.tok_emb.weight

        self.max_len = max_len
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        """Run the forward pass of the language model.

        Args:
            input_ids: (B, T) sequence of vocab indices; 0 reserved for padding/ignore.
            attention_mask: (B, T) optional mask; 1 means valid, 0 means ignored.
            labels: (B, T) optional targets; if provided, returns the cross-entropy loss.

        Returns:
            A dict with the following keys:
            - "logits": (B, T, V) per-position vocabulary distribution.
            - "last_hidden_state": (B, T, C) final-layer hidden representation.
            - "loss": (scalar or None) training loss if labels were provided.
        """
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attention_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # use standard ignore_index for padded/invalid labels
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "last_hidden_state": x, "loss": loss}


class GPTClassificationHead(nn.Module):
    """Classify by passing the last hidden state through a linear layer and dropout."""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Run the classification forward pass.

        Args:
            last_hidden_state: (B, T, C) encoded final hidden representation.
            attention_mask: (B, T) optional mask; unused here.

        Note:
            Uses the last-position token representation as the sentence vector (mean pooling is also possible).
        """
        x = last_hidden_state[:, -1, :]
        x = self.drop(x)
        logits = self.fc(x)
        return logits
