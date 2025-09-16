from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable

import datasets
from torch.utils.data import DataLoader, Dataset

from ..configs import DataConfig
from ..registry import DATASETS
from .text_bpe import BPEBuilder
from .collators import LMTrainCollator, ClassificationCollator


@dataclass
class DatasetBundle:
    train: Optional[Dataset]
    valid: Optional[Dataset]
    test: Optional[Dataset]
    tokenizer  : Any
    collator   : Any
    num_labels : Optional[int] = None


def _text_iter(ds) -> Iterable[str]:
    for rec in ds:
        yield rec.get("text", "")


@DATASETS.register("bookcorpusopen")
def load_bookcorpusopen(cfg: DataConfig) -> DatasetBundle:
    """加载 BookCorpusOpen 并可选训练 BPE。"""
    raw = datasets.load_dataset("bookcorpusopen", split="train")
    builder = None
    if cfg.bpe and cfg.bpe.get("train", False):
        builder = BPEBuilder(cfg.bpe["save_dir"], cfg.bpe["vocab_size"], cfg.bpe["min_freq"])
        builder.train(_text_iter(raw))
        tok = builder.load()
    else:
        builder = BPEBuilder("runs/bpe_default", 40000, 2)
        tok = builder.load()
    collator = LMTrainCollator(tok, seq_len=cfg.seq_len or 512)
    return DatasetBundle(train=raw, valid=None, test=None, tokenizer=tok, collator=collator)


@DATASETS.register("glue")
def load_glue(cfg: DataConfig) -> DatasetBundle:
    """GLUE 多任务加载（MNLI/SST2/...）"""
    task = cfg.task or "mnli"
    raw = datasets.load_dataset("glue", task)
    # 确定字段
    if task in ("sst2",):
        text_cols = ("sentence", None)
        num_labels = 2
    elif task in ("mnli",):
        text_cols = ("premise", "hypothesis")
        num_labels = 3
    elif task in ("mrpc", "qqp"):
        text_cols = ("sentence1", "sentence2")
        num_labels = 2
    else:
        # 对其他任务可继续扩展
        text_cols = ("sentence", None)
        num_labels = 2

    builder = BPEBuilder("runs/bpe_glue", 40000, 2)
    tok = builder.load()
    collator = ClassificationCollator(tok, max_len=cfg.max_len or 256, text_cols=text_cols)
    return DatasetBundle(train=raw.get("train"), valid=raw.get("validation_matched") or raw.get("validation"),
                         test=raw.get("test_matched") or raw.get("test"), tokenizer=tok, collator=collator,
                         num_labels=num_labels)


@DATASETS.register("race")
def load_race(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("race", "all")
    builder = BPEBuilder("runs/bpe_race", 40000, 2)
    tok = builder.load()
    # 该任务使用四选一，Collator 复用 ClassificationCollator（在 tasks.formatting 中将 doc/question/option 拼接）
    collator = ClassificationCollator(tok, max_len=cfg.max_len or 384, text_cols=("article", "question"))
    return DatasetBundle(train=raw["train"], valid=raw["validation"], test=raw["test"],
                         tokenizer=tok, collator=collator, num_labels=4)


@DATASETS.register("story_cloze")
def load_story_cloze(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("story_cloze", "2016")
    builder = BPEBuilder("runs/bpe_story", 40000, 2)
    tok = builder.load()
    collator = ClassificationCollator(tok, max_len=cfg.max_len or 256, text_cols=("input_sentence_1", "sentence_quiz_1"))
    # StoryCloze 实际上是二选一，需要在 formatting 中构建两个候选
    return DatasetBundle(train=None, valid=raw["validation"], test=raw["test"],
                         tokenizer=tok, collator=collator, num_labels=2)


def load_dataset_factory(cfg: DataConfig) -> DatasetBundle:
    return DATASETS.create(cfg.name, cfg=cfg)
