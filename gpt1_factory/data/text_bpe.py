from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPEBuilder:
    """训练与加载 BPE 分词器（兼容 GPT 风格）。

    中文注释：
        - 训练时使用 Whitespace 预分词，适合英文/空格分隔的语料。
        - 若需要中文，可替换为更合适的预分词器或先做切词。
    """

    def __init__(self, save_dir: str | Path, vocab_size: int = 40000, min_freq: int = 2) -> None:
        self.save_dir = Path(save_dir)
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.save_dir / "bpe.json"

    def train(self, iterator: Iterable[str]) -> Tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_freq, special_tokens=[
            "<unk>", "<s>", "</s>", "<sep>"
        ])
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        tokenizer.save(str(self.tokenizer_path))
        return tokenizer

    def load(self) -> Tokenizer:
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")
        return Tokenizer.from_file(str(self.tokenizer_path))
