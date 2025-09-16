from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..registry import MODELS


class GELU(nn.Module):
    """GELU 激活函数的模块封装。"""
    def forward(self, x):
        """前向计算。

        参数:
            x: 任意形状的输入张量。

        返回:
            与输入同形状的张量, 应用 GELU 激活后结果。
        """
        return F.gelu(x)


class CausalSelfAttention(nn.Module):
    """标准掩码自注意力。"""

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
        """前向计算带因果掩码的自注意力。

        参数:
            x: 形状为 (B, T, C) 的输入序列表示。
            attn_mask: 可选注意力掩码, 形状为 (B, T)。0 表示被遮挡。

        返回:
            形状为 (B, T, C) 的更新后表示。
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
    """Transformer 解码块: LN + 自注意力 + MLP 的残差堆叠。"""
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
        """前向计算一个解码块。

        参数:
            x: (B, T, C) 当前层输入。
            attn_mask: (B, T) 可选注意力掩码。

        返回:
            (B, T, C) 残差更新后的输出。
        """
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


@MODELS.register("gpt_decoder")
class GPTDecoderLM(nn.Module):
    """仅解码器 Transformer 语言模型, 亦可作为分类等下游任务骨干。"""

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
        """前向计算语言模型。

        参数:
            input_ids: (B, T) 词表索引序列, 0 预留给 padding/ignore。
            attention_mask: (B, T) 可选掩码, 1 为有效, 0 为忽略。
            labels: (B, T) 可选标签, 若提供则返回交叉熵损失。

        返回:
            包含以下键的字典:
            - "logits": (B, T, V) 每个位置的词表分布。
            - "last_hidden_state": (B, T, C) 最后一层隐藏表示。
            - "loss": (标量或 None) 若提供 labels 则为训练损失。
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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
        return {"logits": logits, "last_hidden_state": x, "loss": loss}


class GPTClassificationHead(nn.Module):
    """把最后一层的 h_m^ℓ 过线性+dropout 做分类。"""

    def __init__(self, d_model: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """前向分类。

        参数:
            last_hidden_state: (B, T, C) 编码后的最后一层隐藏表示。
            attention_mask: (B, T) 可选掩码, 此处未使用。

        说明:
            取句子末位置的 token 表示作为句向量(也可用均值池化)。
        """
        x = last_hidden_state[:, -1, :]
        x = self.drop(x)
        logits = self.fc(x)
        return logits
