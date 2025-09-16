from __future__ import annotations
from pathlib import Path
from typing import Iterable
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class BPEBuilder:
    def __init__(self, save_dir: str | Path, vocab_size: int = 40000, min_freq: int = 2) -> None:
        self.save_dir = Path(save_dir); self.vocab_size = vocab_size; self.min_freq = min_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.save_dir / "bpe.json"

    def _special_tokens(self) -> list[str]:
        return ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]

    def train(self, iterator: Iterable[str]) -> Tokenizer:
        tok = Tokenizer(BPE(unk_token="<unk>")); tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_freq,
                             special_tokens=self._special_tokens())
        tok.train_from_iterator(iterator, trainer=trainer)
        tok.save(str(self.tokenizer_path))
        return tok

    def load_or_train(self, iterator_if_needed: Iterable[str] | None = None) -> Tokenizer:
        if self.tokenizer_path.exists():
            return Tokenizer.from_file(str(self.tokenizer_path))
        if iterator_if_needed is None:
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}. Provide iterator_if_needed to train.")
        return self.train(iterator_if_needed)
