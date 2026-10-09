from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from ..tokenization import GPT1SpecialTokens, TokenizerCompatibilityError


class BPEBuilder:
    def __init__(
        self,
        save_dir: str | Path,
        vocab_size: int = 40000,
        min_freq: int = 2,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = self.save_dir / "bpe.json"
        self.specials = GPT1SpecialTokens()

    def _special_tokens(self) -> list[str]:
        return self.specials.ordered()

    def train(self, iterator: Iterable[str]) -> Tokenizer:
        tok = Tokenizer(BPE(unk_token=self.specials.unk))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_freq,
            special_tokens=self._special_tokens(),
        )
        tok.train_from_iterator(iterator, trainer=trainer)
        tok.save(str(self.tokenizer_path))
        return tok

    def load_or_train(
        self,
        iterator_if_needed: Iterable[str] | None = None,
    ) -> Tokenizer:
        if self.tokenizer_path.exists():
            tok = Tokenizer.from_file(str(self.tokenizer_path))
            self._validate_special_tokens(tok)
            return tok

        if iterator_if_needed is None:
            raise FileNotFoundError(
                f"Tokenizer not found: {self.tokenizer_path}. "
                "Provide iterator_if_needed to train."
            )
        return self.train(iterator_if_needed)

    def _validate_special_tokens(self, tok: Tokenizer) -> None:
        missing = [
            token
            for token in self._special_tokens()
            if tok.token_to_id(token) is None
        ]
        if missing:
            raise TokenizerCompatibilityError(
                "Existing tokenizer predates the GPT-1 task-token contract and "
                f"is missing {missing}. Remove or migrate {self.tokenizer_path} "
                "before starting a new pretraining run."
            )
