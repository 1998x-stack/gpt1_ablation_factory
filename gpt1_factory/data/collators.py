from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from tokenizers import Tokenizer


def _shift_labels_for_lm(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """构造 LM 辅助任务的 labels（右移一位），pad 用于忽略。"""
    # labels 与 input 对齐为预测下一个 token
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    return labels


class LMTrainCollator:
    """语言模型预训练 Collator：将文本拼接后切为长度为 seq_len 的样本。"""

    def __init__(self, tok: Tokenizer, seq_len: int = 512) -> None:
        self.tok = tok
        self.seq_len = seq_len
        self.pad_id = tok.token_to_id("<pad>") or 0

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [ex.get("text", "") for ex in batch]
        ids: List[int] = []
        eos = self.tok.token_to_id("</s>")
        for t in texts:
            out = self.tok.encode(t)
            ids.extend(out.ids + [eos])

        # 划分为 (seq_len+1) 片段以便构造 (x,y)
        chunks = [
            ids[i : i + self.seq_len + 1] for i in range(0, max(0, len(ids) - self.seq_len - 1), self.seq_len + 1)
        ] or [ids[: self.seq_len + 1]]

        x, y = [], []
        for ch in chunks:
            arr = ch[: self.seq_len + 1]
            if len(arr) < self.seq_len + 1:
                arr = arr + [self.pad_id] * (self.seq_len + 1 - len(arr))
            x.append(arr[:-1])
            y.append(arr[1:])

        input_ids = torch.tensor(x, dtype=torch.long)
        labels = torch.tensor(y, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class ClassificationCollator:
    """下游分类 Collator：按任务字段拼接文本，tokenize 到 max_len。
    可选返回 `labels_lm` 以支持微调阶段的辅助 LM 损失。
    """

    def __init__(
        self,
        tok: Tokenizer,
        max_len: int,
        text_cols: Tuple[str, str | None],
        return_lm_labels: bool = True,
    ) -> None:
        self.tok = tok
        self.max_len = max_len
        self.text_cols = text_cols
        self.return_lm_labels = return_lm_labels
        self.pad_id = tok.token_to_id("<pad>") or 0

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        s1, s2 = self.text_cols
        ids_list, attn_list, labels = [], [], []
        for ex in batch:
            t1 = ex[s1] if s1 else ""
            t2 = ex[s2] if s2 else None
            text = f"<s> {t1} <sep> {t2}" if t2 else f"<s> {t1}"
            out = self.tok.encode(text)
            ids = out.ids[: self.max_len]
            attn = [1] * len(ids)
            if len(ids) < self.max_len:
                pad = [self.pad_id] * (self.max_len - len(ids))
                ids, attn = ids + pad, attn + [0] * len(pad)
            ids_list.append(ids)
            attn_list.append(attn)
            if "label" in ex and ex["label"] != -1:
                labels.append(ex["label"])

        input_ids = torch.tensor(ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attn_list, dtype=torch.long)
        batch_out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels:
            batch_out["labels"] = torch.tensor(labels, dtype=torch.long)
        if self.return_lm_labels:
            batch_out["labels_lm"] = _shift_labels_for_lm(input_ids, self.pad_id)
        return batch_out


class MultiChoiceCollator:
    """多选题 Collator（RACE、StoryCloze 等）。

    将每个样本展开为 (num_choices) 个输入，打包成 (B, C, L)，Trainer 负责展平送入骨干网络。
    """

    def __init__(
        self,
        tok: Tokenizer,
        max_len: int,
        options_extractor: Callable[[Dict[str, Any]], List[str]],
        build_inputs: Callable[[Dict[str, Any], str], str],
        label_extractor: Callable[[Dict[str, Any]], int],
        return_lm_labels: bool = True,
    ) -> None:
        self.tok = tok
        self.max_len = max_len
        self.options_extractor = options_extractor
        self.build_inputs = build_inputs
        self.label_extractor = label_extractor
        self.return_lm_labels = return_lm_labels
        self.pad_id = tok.token_to_id("<pad>") or 0

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        input_ids, attn_masks, labels, labels_lm = [], [], [], []

        for ex in batch:
            options = self.options_extractor(ex)
            ids_per, attn_per, lm_per = [], [], []
            for choice in options:
                text = self.build_inputs(ex, choice)
                out = self.tok.encode(text)
                ids = out.ids[: self.max_len]
                attn = [1] * len(ids)
                if len(ids) < self.max_len:
                    pad = [self.pad_id] * (self.max_len - len(ids))
                    ids, attn = ids + pad, attn + [0] * len(pad)
                ids_per.append(ids)
                attn_per.append(attn)
                if self.return_lm_labels:
                    lm_per.append(_shift_labels_for_lm(torch.tensor([ids]), self.pad_id).squeeze(0).tolist())
            input_ids.append(ids_per)
            attn_masks.append(attn_per)
            if "label" in ex and ex["label"] != -1:
                labels.append(self.label_extractor(ex))
            if self.return_lm_labels:
                labels_lm.append(lm_per)

        input_ids = torch.tensor(input_ids, dtype=torch.long)        # (B, C, L)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long)  # (B, C, L)
        batch_out = {"input_ids": input_ids, "attention_mask": attention_mask}

        if labels:
            batch_out["labels"] = torch.tensor(labels, dtype=torch.long)
        if self.return_lm_labels:
            batch_out["labels_lm"] = torch.tensor(labels_lm, dtype=torch.long)
        return batch_out
