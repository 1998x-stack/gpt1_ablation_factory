from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import datasets
from torch.utils.data import Dataset

from ..configs import DataConfig
from ..registry import DATASETS
from .collators import (
    ClassificationCollator,
    LMTrainCollator,
    MultiChoiceCollator,
    SimilarityCollator,
)
from .text_bpe import BPEBuilder


@dataclass
class DatasetBundle:
    train: Optional[Dataset]
    valid: Optional[Dataset]
    test: Optional[Dataset]
    tokenizer: Any
    collator: Any
    num_labels: Optional[int] = None


def _text_iter(
    ds,
    cols: Tuple[str, str | None] | None = None,
) -> Iterable[str]:
    if ds is None:
        return []
    if cols is None:
        for rec in ds:
            yield rec.get("text", "")
    else:
        a, b = cols
        for rec in ds:
            t1 = rec[a] if a else ""
            t2 = rec[b] if b else ""
            yield f"{t1}\n{t2}"


def _load_books_like_split(cfg: DataConfig) -> datasets.Dataset:
    """Try several long-text corpora and local .txt, using cache_dir."""

    cache_dir = cfg.cache_dir
    if cfg.local_text_dir and Path(cfg.local_text_dir).exists():
        files = glob(
            os.path.join(cfg.local_text_dir, "**/*.txt"),
            recursive=True,
        )
        if files:
            return datasets.load_dataset(
                "text",
                data_files={"train": files},
                cache_dir=cache_dir,
            )["train"]

    try:
        return datasets.load_dataset(
            "hf://datasets/bookcorpusopen/bookcorpusopen",
            split="train",
            cache_dir=cache_dir,
        )
    except Exception:
        pass

    for cand in (
        "Skylion007/openwebtext",
        "hf://datasets/Skylion007/openwebtext",
    ):
        try:
            return datasets.load_dataset(
                cand,
                split="train",
                cache_dir=cache_dir,
            )
        except Exception:
            continue

    for cfg_name in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            return datasets.load_dataset(
                "wikitext",
                cfg_name,
                split="train",
                cache_dir=cache_dir,
            )
        except Exception:
            continue

    return datasets.load_dataset(
        "ag_news",
        split="train",
        cache_dir=cache_dir,
    )


def resolve_model_vocab_size(tokenizer: Any, configured: int) -> int:
    tok_vocab = tokenizer.get_vocab_size()
    if (
        isinstance(configured, int)
        and configured > 0
        and configured != tok_vocab
    ):
        warnings.warn(
            f"Configured vocab_size ({configured}) differs from the tokenizer "
            f"vocabulary size ({tok_vocab}); using {tok_vocab}."
        )
    return tok_vocab


def _require_inherited_tokenizer(tokenizer: Any, dataset_name: str):
    if tokenizer is None:
        raise ValueError(
            f"{dataset_name} requires the tokenizer artifact inherited from "
            "pretraining; downstream tokenizer retraining is disabled."
        )
    return tokenizer


@DATASETS.register("local_text")
def load_local_text(cfg: DataConfig) -> DatasetBundle:
    text_dir = Path(
        cfg.local_text_dir or "gpt1_ablation_factory/data/text"
    )
    if not text_dir.exists():
        raise FileNotFoundError(
            f"local_text corpus dir not found: {text_dir}"
        )
    files = sorted(
        glob(str(text_dir / "**/*.txt"), recursive=True)
    )
    if not files:
        raise FileNotFoundError(
            f"No *.txt files found in local_text_dir: {text_dir}"
        )

    raw = datasets.load_dataset(
        "text",
        data_files={"train": files},
        cache_dir=cfg.cache_dir,
    )["train"]

    bpe_cfg = cfg.bpe or {}
    builder = BPEBuilder(
        bpe_cfg.get("save_dir", "runs/bpe_local"),
        bpe_cfg.get("vocab_size", 4000),
        bpe_cfg.get("min_freq", 2),
    )
    tok = builder.load_or_train(_text_iter(raw, None))
    collator = LMTrainCollator(
        tok,
        seq_len=cfg.seq_len or 256,
    )
    return DatasetBundle(
        train=raw,
        valid=None,
        test=None,
        tokenizer=tok,
        collator=collator,
    )


@DATASETS.register("bookcorpusopen")
def load_bookcorpusopen(cfg: DataConfig) -> DatasetBundle:
    try:
        raw = datasets.load_dataset(
            "bookcorpusopen",
            split="train",
            cache_dir=cfg.cache_dir,
        )
    except Exception:
        raw = _load_books_like_split(cfg)

    builder = BPEBuilder(
        (
            cfg.bpe["save_dir"]
            if cfg.bpe and "save_dir" in cfg.bpe
            else "runs/bpe_bookscorpus"
        ),
        cfg.bpe.get("vocab_size", 40000) if cfg.bpe else 40000,
        cfg.bpe.get("min_freq", 2) if cfg.bpe else 2,
    )
    tok = builder.load_or_train(_text_iter(raw))
    collator = LMTrainCollator(
        tok,
        seq_len=cfg.seq_len or 512,
    )
    return DatasetBundle(
        train=raw,
        valid=None,
        test=None,
        tokenizer=tok,
        collator=collator,
    )


@DATASETS.register("glue")
def load_glue(
    cfg: DataConfig,
    tokenizer=None,
) -> DatasetBundle:
    tok = _require_inherited_tokenizer(tokenizer, "GLUE")
    task = cfg.task or "mnli"
    raw = datasets.load_dataset(
        "glue",
        task,
        cache_dir=cfg.cache_dir,
    )

    if task == "sst2":
        text_cols, num_labels = ("sentence", None), 2
        collator = ClassificationCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=text_cols,
            return_lm_labels=True,
            mode="classification",
        )
    elif task == "mnli":
        text_cols, num_labels = ("premise", "hypothesis"), 3
        collator = ClassificationCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=text_cols,
            return_lm_labels=True,
            mode="entailment",
        )
    elif task in ("mrpc", "qqp"):
        text_cols, num_labels = ("sentence1", "sentence2"), 2
        collator = SimilarityCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=(text_cols[0], text_cols[1]),
            return_lm_labels=True,
        )
    elif task == "stsb":
        text_cols, num_labels = ("sentence1", "sentence2"), 1
        collator = SimilarityCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=(text_cols[0], text_cols[1]),
            return_lm_labels=True,
            regression=True,
        )
    elif task == "cola":
        text_cols, num_labels = ("sentence", None), 2
        collator = ClassificationCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=text_cols,
            return_lm_labels=True,
            mode="classification",
        )
    else:
        text_cols, num_labels = ("sentence", None), 2
        collator = ClassificationCollator(
            tok,
            max_len=cfg.max_len or 256,
            text_cols=text_cols,
            return_lm_labels=True,
            mode="classification",
        )

    valid_split = (
        raw["validation_matched"]
        if "validation_matched" in raw
        else raw.get("validation")
    )
    test_split = (
        raw["test_matched"]
        if "test_matched" in raw
        else raw.get("test")
    )
    return DatasetBundle(
        train=raw.get("train"),
        valid=valid_split,
        test=test_split,
        tokenizer=tok,
        collator=collator,
        num_labels=num_labels,
    )


@DATASETS.register("race")
def load_race(
    cfg: DataConfig,
    tokenizer=None,
) -> DatasetBundle:
    tok = _require_inherited_tokenizer(tokenizer, "RACE")
    raw = datasets.load_dataset(
        "race",
        "all",
        cache_dir=cfg.cache_dir,
    )

    def options_extractor(ex) -> List[str]:
        return ex["options"]

    def context_extractor(ex) -> str:
        return ex["article"]

    def question_extractor(ex) -> str:
        return ex["question"]

    def label_extractor(ex) -> int:
        return "ABCD".index(ex["answer"])

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 384,
        options_extractor=options_extractor,
        context_extractor=context_extractor,
        question_extractor=question_extractor,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(
        train=raw["train"],
        valid=raw["validation"],
        test=raw["test"],
        tokenizer=tok,
        collator=collator,
        num_labels=4,
    )


@DATASETS.register("story_cloze")
def load_story_cloze(
    cfg: DataConfig,
    tokenizer=None,
) -> DatasetBundle:
    tok = _require_inherited_tokenizer(tokenizer, "StoryCloze")
    raw = datasets.load_dataset(
        "story_cloze",
        "2016",
        cache_dir=cfg.cache_dir,
    )

    def options_extractor(ex) -> List[str]:
        keys = [
            key
            for key in ex.keys()
            if "ending" in key.lower() or "quiz" in key.lower()
        ]
        if len(keys) < 2:
            raise KeyError(
                "StoryCloze ending fields were not found."
            )
        keys = sorted(keys)[:2]
        return [ex[keys[0]], ex[keys[1]]]

    def context_extractor(ex) -> str:
        sentence_keys = sorted(
            key
            for key in ex.keys()
            if "sentence" in key.lower()
        )
        return " ".join(
            str(ex[key])
            for key in sentence_keys[:4]
            if ex.get(key)
        )

    def label_extractor(ex) -> int:
        if "answer_right_ending" in ex:
            return int(ex["answer_right_ending"]) - 1
        return -1

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 256,
        options_extractor=options_extractor,
        context_extractor=context_extractor,
        question_extractor=None,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(
        train=None,
        valid=raw["validation"],
        test=raw["test"],
        tokenizer=tok,
        collator=collator,
        num_labels=2,
    )


def load_dataset_factory(
    cfg: DataConfig,
    tokenizer=None,
) -> DatasetBundle:
    return DATASETS.create(
        cfg.name,
        cfg=cfg,
        tokenizer=tokenizer,
    )
