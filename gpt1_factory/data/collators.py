from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer


class LMTrainCollator:
    """语言模型预训练 Collator：将长文本切成固定 seq_len 的片段。"""

    def __init__(self, tok: Tokenizer, seq_len: int = 512) -> None:
        self.tok = tok
        self.seq_len = seq_len

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex.get("text", "") for ex in batch]
        ids = []
        for t in texts:
            out = self.tok.encode(t)
            ids.extend(out.ids + [self.tok.token_to_id("</s>")])
        # 切块
        chunks = [ids[i : i + self.seq_len + 1] for i in range(0, max(0, len(ids) - self.seq_len - 1), self.seq_len + 1)]
        if not chunks:
            chunks = [ids[: self.seq_len + 1]]
        x = []
        y = []
        for ch in chunks:
            arr = ch[: self.seq_len + 1]
            if len(arr) < self.seq_len + 1:
                arr = arr + [self.tok.token_to_id("<pad>")] * (self.seq_len + 1 - len(arr))
            x.append(arr[:-1])
            y.append(arr[1:])
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long),
            "attention_mask": torch.ones((len(x), self.seq_len), dtype=torch.long),
        }


class ClassificationCollator:
    """下游分类 Collator：按任务字段拼接文本，tokenize 到 max_len。"""

    def __init__(self, tok: Tokenizer, max_len: int, text_cols: Tuple[str, str | None]) -> None:
        self.tok = tok
        self.max_len = max_len
        self.text_cols = text_cols

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        s1, s2 = self.text_cols
        encoded = []
        labels = []
        for ex in batch:
            t1 = ex[s1] if s1 else ""
            t2 = ex[s2] if s2 else None
            text = f"<s> {t1} <sep> {t2}" if t2 else f"<s> {t1}"
            out = self.tok.encode(text)
            ids = out.ids[: self.max_len]
            attn = [1] * len(ids)
            if len(ids) < self.max_len:
                pad_id = self.tok.token_to_id("<pad>")
                pad = [pad_id] * (self.max_len - len(ids))
                ids = ids + pad
                attn = attn + [0] * len(pad)
            encoded.append((ids, attn))
            if "label" in ex and ex["label"] != -1:
                labels.append(ex["label"])

        input_ids = torch.tensor([ids for ids, _ in encoded], dtype=torch.long)
        attention_mask = torch.tensor([attn for _, attn in encoded], dtype=torch.long)
        batch_out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels:
            batch_out["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch_out
