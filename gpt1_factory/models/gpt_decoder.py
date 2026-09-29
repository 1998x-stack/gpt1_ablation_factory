from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..registry import MODELS


class GELU(nn.Module):
    """Module wrapper around the GELU activation."""

    def forward(self, x):
        return F.gelu(x)


class CausalSelfAttention(nn.Module):
    """Causal masked self-attention."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_dropout: float,
        resid_dropout: float,
        max_len: int,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.d_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).view(
            1, 1, max_len, max_len
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            rearrange(t, "b t (h d) -> b h t d", h=self.n_head)
            for t in qkv
        ]

        att = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        causal = self.mask[:, :, :T, :T]
        att = att.masked_fill(~causal, float("-inf"))

        if attn_mask is not None:
            att = att.masked_fill(
                attn_mask[:, None, None, :T] == 0,
                float("-inf"),
            )

        att = self.attn_drop(att.softmax(dim=-1))
        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.resid_drop(self.d_proj(y))


class Block(nn.Module):
    """Transformer block supporting GPT-1 Post-LN and modern Pre-LN."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        resid_dropout: float,
        max_len: int,
        layer_norm_eps: float = 1e-5,
        norm_style: str = "post_ln",
    ) -> None:
        super().__init__()
        if norm_style not in {"post_ln", "pre_ln"}:
            raise ValueError(
                f"Unsupported norm_style={norm_style!r}; expected 'post_ln' or 'pre_ln'."
            )

        self.norm_style = norm_style
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.attn = CausalSelfAttention(
            d_model,
            n_head,
            attn_dropout,
            resid_dropout,
            max_len,
        )
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.norm_style == "post_ln":
            x = self.ln1(x + self.attn(x, attn_mask))
            x = self.ln2(x + self.mlp(x))
            return x

        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


@MODELS.register("gpt_decoder")
class GPTDecoderLM(nn.Module):
    """Decoder-only Transformer with explicit GPT-1 fidelity switches."""

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
        tie_emb: bool = True,
        layer_norm_eps: float = 1e-5,
        gelu: bool = True,
        norm_style: str = "post_ln",
        final_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        del gelu  # retained for config compatibility; GPT-1 uses GELU.

        self.d_model = d_model
        self.max_len = max_len
        self.norm_style = norm_style
        self.final_layer_norm = final_layer_norm
        self.tie_emb = tie_emb

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_head=n_head,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    max_len=max_len,
                    layer_norm_eps=layer_norm_eps,
                    norm_style=norm_style,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f: nn.Module = (
            nn.LayerNorm(d_model, eps=layer_norm_eps)
            if final_layer_norm
            else nn.Identity()
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        if tie_emb:
            # GPT-1 Eq. (2): language-model projection reuses W_e^T.
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds model max_len={self.max_len}."
            )

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "last_hidden_state": x,
            "loss": loss,
        }


class GPTClassificationHead(nn.Module):
    """Backward-compatible classification head using the last valid token."""

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
    ):
        if attention_mask is None:
            x = last_hidden_state[:, -1, :]
        else:
            positions = (
                torch.arange(
                    last_hidden_state.size(1),
                    device=last_hidden_state.device,
                )
                .unsqueeze(0)
                .expand_as(attention_mask)
                .masked_fill(attention_mask == 0, -1)
                .max(dim=-1)
                .values
                .clamp_min(0)
            )
            x = last_hidden_state[
                torch.arange(last_hidden_state.size(0), device=x_device := last_hidden_state.device),
                positions,
            ]
        return self.fc(self.drop(x))
