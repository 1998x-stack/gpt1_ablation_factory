from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, List

import datasets
from torch.utils.data import DataLoader, Dataset

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
    if cols is None:
        for rec in ds:
            yield rec.get("text", "")
    else:
        a, b = cols
        for rec in ds:
            t1 = rec[a] if a else ""
            t2 = rec[b] if b else ""
            yield f"{t1}\n{t2}"


@DATASETS.register("bookcorpusopen")
def load_bookcorpusopen(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("bookcorpusopen", split="train")
    builder = BPEBuilder(cfg.bpe["save_dir"], cfg.bpe["vocab_size"], cfg.bpe["min_freq"]) if cfg.bpe else BPEBuilder(
        "runs/bpe_bookscorpus", 40000, 2
    )
    tok = builder.load_or_train(_text_iter(raw))
    collator = LMTrainCollator(tok, seq_len=cfg.seq_len or 512)
    return DatasetBundle(train=raw, valid=None, test=None, tokenizer=tok, collator=collator)


@DATASETS.register("glue")
def load_glue(cfg: DataConfig) -> DatasetBundle:
    task = cfg.task or "mnli"
    raw = datasets.load_dataset("glue", task)

    # 字段与标签映射
    if task == "sst2":
        text_cols, num_labels = ("sentence", None), 2
    elif task == "mnli":
        text_cols, num_labels = ("premise", "hypothesis"), 3
    elif task in ("mrpc", "qqp"):
        text_cols, num_labels = ("sentence1", "sentence2"), 2
    elif task == "cola":
        text_cols, num_labels = ("sentence", None), 2
    elif task == "stsb":
        text_cols, num_labels = ("sentence1", "sentence2"), 1  # 回归
    else:
        text_cols, num_labels = ("sentence", None), 2

    builder = BPEBuilder("runs/bpe_glue", 40000, 2)
    tok = builder.load_or_train(_text_iter(raw["train"], text_cols))
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
        # 'answer' in {'A','B','C','D'}
        return "ABCD".index(ex["answer"])

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 384,
        options_extractor=options_extractor,
        build_inputs=build_inputs,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(
        train=raw["train"], valid=raw["validation"], test=raw["test"], tokenizer=tok, collator=collator, num_labels=4
    )


@DATASETS.register("story_cloze")
def load_story_cloze(cfg: DataConfig) -> DatasetBundle:
    raw = datasets.load_dataset("story_cloze", "2016")
    # 字段名在该数据集中有一定差异，这里仅使用第二阶段验证/测试（官方不提供train）
    builder = BPEBuilder("runs/bpe_story", 40000, 2)
    # 用 validation 近似训练分词器
    tok = builder.load_or_train(_text_iter(raw["validation"], None))

    def options_extractor(ex) -> List[str]:
        # 尝试通用键名（不同清洗版本字段名略有不同）；若不存在请按你的本地字段微调
        keys = [k for k in ex.keys() if "ending" in k or "quiz" in k]
        if len(keys) >= 2:
            # 取两个候选，按键名排序保证稳定
            keys = sorted(keys)[:2]
            return [ex[keys[0]], ex[keys[1]]]
        # 兜底（不可用时抛错以便用户修正）
        raise KeyError("StoryCloze sample lacks recognizable ending fields (e.g., 'ending1','ending2').")

    def build_inputs(ex, choice: str) -> str:
        # 拼接前四句 + 备选结尾
        sents = [ex.get(k) for k in ex.keys() if "sentence" in k.lower()]
        prefix = " ".join(sents[:4]) if sents else ""
        return f"<s> {prefix} <sep> {choice}"

    def label_extractor(ex) -> int:
        # 数据集中通常包含正确结尾索引（1/2）；若不存在，标注会缺失（只做评估）
        if "answer_right_ending" in ex:
            # 1/2 → 0/1
            return int(ex["answer_right_ending"]) - 1
        return -1  # 允许无标签的验证/测试

    collator = MultiChoiceCollator(
        tok,
        max_len=cfg.max_len or 256,
        options_extractor=options_extractor,
        build_inputs=build_inputs,
        label_extractor=label_extractor,
        return_lm_labels=True,
    )
    return DatasetBundle(train=None, valid=raw["validation"], test=raw["test"], tokenizer=tok, collator=collator, num_labels=2)


def load_dataset_factory(cfg: DataConfig) -> DatasetBundle:
    return DATASETS.create(cfg.name, cfg=cfg)
