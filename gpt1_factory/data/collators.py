from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from ..tasks.protocol import GPT1TaskProtocol, TaskEncoding


_IGNORE_IDX = -100


def _shift_labels_for_lm(
    input_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Right-shift labels for next-token prediction; pad positions -> -100."""

    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_id
    labels[labels == pad_id] = _IGNORE_IDX
    return labels


def _pad_task_encoding(
    encoding: TaskEncoding,
    *,
    max_len: int,
    pad_id: int,
) -> tuple[list[int], list[int], int]:
    ids = list(encoding.input_ids)
    if len(ids) > max_len:
        raise ValueError(
            "Task protocol returned a sequence longer than max_len; "
            "protocol truncation invariant was violated."
        )
    attn = [1] * len(ids)
    if len(ids) < max_len:
        n_pad = max_len - len(ids)
        ids.extend([pad_id] * n_pad)
        attn.extend([0] * n_pad)
    return ids, attn, encoding.classify_position


class LMTrainCollator:
    """LM pretraining collator: pack text and cut into fixed seq_len blocks."""

    def __init__(self, tok: Tokenizer, seq_len: int = 512) -> None:
        self.tok = tok
        self.seq_len = seq_len
        self.pad_id = tok.token_to_id("<pad>")
        self.eos_id = tok.token_to_id("</s>")
        if self.pad_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must contain <pad> and </s> for LM batching.")

    def __call__(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        texts = [ex.get("text", "") for ex in batch]
        ids: List[int] = []
        for text in texts:
            out = self.tok.encode(text, add_special_tokens=False)
            ids.extend(out.ids + [self.eos_id])

        chunks = [
            ids[i : i + self.seq_len + 1]
            for i in range(
                0,
                max(0, len(ids) - self.seq_len - 1),
                self.seq_len + 1,
            )
        ] or [ids[: self.seq_len + 1]]

        x, y = [], []
        for chunk in chunks:
            arr = chunk[: self.seq_len + 1]
            if len(arr) < self.seq_len + 1:
                arr = arr + [self.pad_id] * (self.seq_len + 1 - len(arr))
            x.append(arr[:-1])
            y.append(arr[1:])

        input_ids = torch.tensor(x, dtype=torch.long)
        labels = torch.tensor(y, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        labels[labels == self.pad_id] = _IGNORE_IDX
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class ClassificationCollator:
    """GPT-1 classification/entailment collator with explicit classify positions."""

    def __init__(
        self,
        tok: Tokenizer,
        max_len: int,
        text_cols: Tuple[str, str | None],
        return_lm_labels: bool = True,
        mode: str = "classification",
        regression: bool = False,
    ) -> None:
        if mode not in {"classification", "entailment"}:
            raise ValueError(
                "ClassificationCollator mode must be 'classification' or 'entailment'."
            )
        self.tok = tok
        self.max_len = max_len
        self.text_cols = text_cols
        self.return_lm_labels = return_lm_labels
        self.mode = mode
        self.regression = regression
        self.pad_id = tok.token_to_id("<pad>")
        if self.pad_id is None:
            raise ValueError("Tokenizer must contain <pad>.")
        self.protocol = GPT1TaskProtocol(tok, max_len=max_len)

    def __call__(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        s1, s2 = self.text_cols
        ids_list: list[list[int]] = []
        attn_list: list[list[int]] = []
        positions: list[int] = []
        label_values: list[Any] = []
        all_labeled = True

        for ex in batch:
            first = str(ex[s1]) if s1 else ""
            second = str(ex[s2]) if s2 else None

            if self.mode == "entailment":
                if second is None:
                    raise ValueError("Entailment protocol requires two text fields.")
                encoding = self.protocol.encode_entailment(first, second)
            else:
                encoding = self.protocol.encode_classification(first)

            ids, attn, classify_position = _pad_task_encoding(
                encoding,
                max_len=self.max_len,
                pad_id=self.pad_id,
            )
            ids_list.append(ids)
            attn_list.append(attn)
            positions.append(classify_position)

            label = ex.get("label", -1)
            if label == -1:
                all_labeled = False
            else:
                label_values.append(label)

        input_ids = torch.tensor(ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attn_list, dtype=torch.long)
        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "classify_positions": torch.tensor(positions, dtype=torch.long),
        }

        if all_labeled and len(label_values) == len(batch):
            dtype = torch.float32 if self.regression else torch.long
            batch_out["labels"] = torch.tensor(label_values, dtype=dtype)

        if self.return_lm_labels:
            batch_out["labels_lm"] = _shift_labels_for_lm(
                input_ids,
                self.pad_id,
            )
        return batch_out


class SimilarityCollator:
    """Symmetric pair protocol: encode A->B and B->A for every example."""

    def __init__(
        self,
        tok: Tokenizer,
        max_len: int,
        text_cols: Tuple[str, str],
        return_lm_labels: bool = True,
        regression: bool = False,
    ) -> None:
        self.tok = tok
        self.max_len = max_len
        self.text_cols = text_cols
        self.return_lm_labels = return_lm_labels
        self.regression = regression
        self.pad_id = tok.token_to_id("<pad>")
        if self.pad_id is None:
            raise ValueError("Tokenizer must contain <pad>.")
        self.protocol = GPT1TaskProtocol(tok, max_len=max_len)

    def __call__(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        first_col, second_col = self.text_cols
        ids_batch: list[list[list[int]]] = []
        attn_batch: list[list[list[int]]] = []
        pos_batch: list[list[int]] = []
        label_values: list[Any] = []
        all_labeled = True

        for ex in batch:
            encodings = self.protocol.encode_similarity(
                str(ex[first_col]),
                str(ex[second_col]),
            )
            ids_pair: list[list[int]] = []
            attn_pair: list[list[int]] = []
            pos_pair: list[int] = []

            for encoding in encodings:
                ids, attn, classify_position = _pad_task_encoding(
                    encoding,
                    max_len=self.max_len,
                    pad_id=self.pad_id,
                )
                ids_pair.append(ids)
                attn_pair.append(attn)
                pos_pair.append(classify_position)

            ids_batch.append(ids_pair)
            attn_batch.append(attn_pair)
            pos_batch.append(pos_pair)

            label = ex.get("label", -1)
            if label == -1:
                all_labeled = False
            else:
                label_values.append(label)

        input_ids = torch.tensor(ids_batch, dtype=torch.long)
        attention_mask = torch.tensor(attn_batch, dtype=torch.long)
        batch_out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "classify_positions": torch.tensor(pos_batch, dtype=torch.long),
            "similarity_pairs": torch.tensor(True),
        }

        if all_labeled and len(label_values) == len(batch):
            dtype = torch.float32 if self.regression else torch.long
            batch_out["labels"] = torch.tensor(label_values, dtype=dtype)

        if self.return_lm_labels:
            B, C, L = input_ids.shape
            lm = _shift_labels_for_lm(
                input_ids.view(B * C, L),
                self.pad_id,
            ).view(B, C, L)
            batch_out["labels_lm"] = lm

        return batch_out


class MultiChoiceCollator:
    """GPT-1 multiple-choice protocol yielding candidate traversals (B, C, L)."""

    def __init__(
        self,
        tok: Tokenizer,
        max_len: int,
        options_extractor: Callable[[Dict[str, Any]], List[str]],
        context_extractor: Callable[[Dict[str, Any]], str],
        label_extractor: Callable[[Dict[str, Any]], int],
        question_extractor: Callable[[Dict[str, Any]], str | None] | None = None,
        return_lm_labels: bool = True,
    ) -> None:
        self.tok = tok
        self.max_len = max_len
        self.options_extractor = options_extractor
        self.context_extractor = context_extractor
        self.question_extractor = question_extractor
        self.label_extractor = label_extractor
        self.return_lm_labels = return_lm_labels
        self.pad_id = tok.token_to_id("<pad>")
        if self.pad_id is None:
            raise ValueError("Tokenizer must contain <pad>.")
        self.protocol = GPT1TaskProtocol(tok, max_len=max_len)

    def __call__(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        input_ids: list[list[list[int]]] = []
        attn_masks: list[list[list[int]]] = []
        classify_positions: list[list[int]] = []
        labels: list[int] = []
        all_labeled = True

        for ex in batch:
            context = str(self.context_extractor(ex))
            question = (
                self.question_extractor(ex)
                if self.question_extractor is not None
                else None
            )
            options = self.options_extractor(ex)
            ids_per: list[list[int]] = []
            attn_per: list[list[int]] = []
            pos_per: list[int] = []

            for choice in options:
                encoding = self.protocol.encode_multiple_choice(
                    context,
                    None if question is None else str(question),
                    str(choice),
                )
                ids, attn, classify_position = _pad_task_encoding(
                    encoding,
                    max_len=self.max_len,
                    pad_id=self.pad_id,
                )
                ids_per.append(ids)
                attn_per.append(attn)
                pos_per.append(classify_position)

            input_ids.append(ids_per)
            attn_masks.append(attn_per)
            classify_positions.append(pos_per)

            label = int(self.label_extractor(ex))
            if label < 0:
                all_labeled = False
            else:
                labels.append(label)

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long)
        batch_out = {
            "input_ids": input_ids_t,
            "attention_mask": attention_mask,
            "classify_positions": torch.tensor(
                classify_positions,
                dtype=torch.long,
            ),
        }

        if all_labeled and len(labels) == len(batch):
            batch_out["labels"] = torch.tensor(labels, dtype=torch.long)

        if self.return_lm_labels:
            B, C, L = input_ids_t.shape
            batch_out["labels_lm"] = _shift_labels_for_lm(
                input_ids_t.view(B * C, L),
                self.pad_id,
            ).view(B, C, L)

        return batch_out
