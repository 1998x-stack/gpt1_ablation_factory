from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, List
import os
from glob import glob

import datasets
from torch.utils.data import Dataset

from ..configs import DataConfig
from ..registry import DATASETS
from .text_bpe import BPEBuilder
from .collators import LMTrainCollator, ClassificationCollator, MultiChoiceCollator


@dataclass
class DatasetBundle:
    train: Optional[Dataset]
    valid: Optional[Dataset]
    test: Optional[Dataset]
    tokenizer: Any
    collator: Any
    num_labels: Optional[int] = None


def _text_iter(ds, cols: Tuple[str, str | None] | None = None) -> Iterable[str]:
    """从数据集中提取用于 BPE 训练的文本迭代器。"""
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


def _load_books_like_split() -> datasets.Dataset:
    """
    在新版本 datasets 中，脚本式 community dataset 会被禁用。
    我们做一个‘多级回退’：
      1) 标准 hub path（新风格地址）
      2) OpenWebText（近似书籍/长文本）
      3) WikiText-103 raw（官方维护）
      4) 本地文本目录（环境变量 BOOKS_TEXT_DIR 下的 *.txt）
    """
    # 1) 尝试 hub path（新风格地址）
    try:
        # 某些镜像将数据托管为 parquet/arrow，无需执行脚本
        return datasets.load_dataset("hf://datasets/bookcorpusopen/bookcorpusopen", split="train")
    except Exception:
        pass

    # 2) OpenWebText（Skylion007/openwebtext）
    for cand in ("Skylion007/openwebtext", "hf://datasets/Skylion007/openwebtext"):
        try:
            return datasets.load_dataset(cand, split="train")
        except Exception:
            continue

    # 3) WikiText（官方）
    for cfg in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        for cand in ("wikitext", f"hf://datasets/wikitext/{cfg}"):
            try:
                return datasets.load_dataset("wikitext", cfg, split="train")
            except Exception:
                continue

    # 4) 本地文本目录
    local_dir = os.environ.get("BOOKS_TEXT_DIR", "").strip()
    if local_dir:
        files = glob(os.path.join(local_dir, "**/*.txt"), recursive=True)
        if not files:
            raise FileNotFoundError(f"BOOKS_TEXT_DIR={local_dir} 下未找到 *.txt")
        return datasets.load_dataset("text", data_files={"train": files})["train"]

    # 实在不行，退到 AG News 作为占位（可运行）
    return datasets.load_dataset("ag_news", split="train")


@DATASETS.register("bookcorpusopen")
def load_bookcorpusopen(cfg: DataConfig) -> DatasetBundle:
    # 新版 datasets 不再支持远程脚本，走回退加载
    try:
        raw = datasets.load_dataset("bookcorpusopen", split="train")
    except Exception:
        raw = _load_books_like_split()

    builder = BPEBuilder(
        cfg.bpe["save_dir"] if (cfg.bpe and "save_dir" in cfg.bpe) else "runs/bpe_bookscorpus",
        cfg.bpe.get("vocab_size", 40000) if cfg.bpe else 40000,
        cfg.bpe.get("min_freq", 2) if cfg.bpe else 2,
    )
    tok = builder.load_or_train(_text_iter(raw))
    collator = LMTrainCollator(tok, seq_len=cfg.seq_len or 512)
    return DatasetBundle(train=raw, valid=None, test=None, tokenizer=tok, collator=collator)


@DATASETS.register("glue")
def load_glue(cfg: DataConfig) -> DatasetBundle:
    task = cfg.task or "mnli"
    raw = datasets.load_dataset("glue", task)

    if task == "sst2":
        text_cols, num_labels = ("sentence", None), 2
    elif task == "mnli":
        text_cols, num_labels = ("premise", "hypothesis"), 3
    elif task in ("mrpc", "qqp"):
        text_cols, num_labels = ("sentence1", "sentence2"), 2
    elif task == "cola":
        text_cols, num_labels = ("sentence", None), 2
    elif task == "stsb":
        text_cols, num_labels = ("sentence1", "sentence2"), 1
    else:
        text_cols, num_labels = ("sentence", None), 2

    builder = BPEBuilder("runs/bpe_glue", 40000, 2)
    tok = builder.load_or_train(_text_iter(raw.get("train"), text_cols))
    collator = ClassificationCollator(tok, max_len=cfg.max_len or 256, text_cols=text_cols, return_lm_labels=True)

    valid_split = raw.get("validation_matched") or raw.get("validation")
    test_split = raw.get("test_matched") or raw.get("test")
    return DatasetBundle(
        train=raw.get("train"),
        valid=valid_split,
        test=test_split,
        tokenizer=tok,
        collator=collator,
        num_labels=num_labels,
    )


@DATASETS.register("race")
def load_race(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("race", "all")
    builder = BPEBuilder("runs/bpe_race", 40000, 2)
    tok = builder.load_or_train(_text_iter(raw["train"], ("article", "question")))

    def options_extractor(ex) -> List[str]:
        return ex["options"]

    def build_inputs(ex, choice: str) -> str:
        return f"<s> {ex['article']} <sep> {ex['question']} <sep> {choice}"

    def label_extractor(ex) -> int:
        return "ABCD".index(ex["answer"])

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 384,
        options_extractor=options_extractor,
        build_inputs=build_inputs,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(train=raw["train"], valid=raw["validation"], test=raw["test"],
                         tokenizer=tok, collator=collator, num_labels=4)


@DATASETS.register("story_cloze")
def load_story_cloze(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("story_cloze", "2016")
    builder = BPEBuilder("runs/bpe_story", 40000, 2)
    tok = builder.load_or_train(_text_iter(raw["validation"], None))

    def options_extractor(ex) -> List[str]:
        keys = [k for k in ex.keys() if "ending" in k or "quiz" in k]
        if len(keys) >= 2:
            keys = sorted(keys)[:2]
            return [ex[keys[0]], ex[keys[1]]]
        raise KeyError("StoryCloze: 未找到候选结尾字段（例如 'ending1','ending2'）。")

    def build_inputs(ex, choice: str) -> str:
        sents = [ex.get(k) for k in ex.keys() if "sentence" in k.lower()]
        prefix = " ".join(sents[:4]) if sents else ""
        return f"<s> {prefix} <sep> {choice}"

    def label_extractor(ex) -> int:
        if "answer_right_ending" in ex:
            return int(ex["answer_right_ending"]) - 1
        return -1

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 256,
        options_extractor=options_extractor,
        build_inputs=build_inputs,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(train=None, valid=raw["validation"], test=raw["test"],
                         tokenizer=tok, collator=collator, num_labels=2)


def load_dataset_factory(cfg: DataConfig) -> DatasetBundle:
    return DATASETS.create(cfg.name, cfg=cfg)
