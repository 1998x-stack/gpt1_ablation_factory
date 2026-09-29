from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT1SpecialTokens:
    """Special tokens used by the GPT-1 task protocol.

    The task tokens mirror the names used by OpenAI's 2018 reference
    implementation. Repository infrastructure keeps explicit pad/unk/eos tokens
    for batching and generation.
    """

    pad: str = "<pad>"
    unk: str = "<unk>"
    eos: str = "</s>"
    start: str = "_start_"
    delimiter: str = "_delimiter_"
    classify: str = "_classify_"

    def ordered(self) -> list[str]:
        return [
            self.pad,
            self.unk,
            self.eos,
            self.start,
            self.delimiter,
            self.classify,
        ]

    def as_dict(self) -> dict[str, str]:
        return {
            "pad": self.pad,
            "unk": self.unk,
            "eos": self.eos,
            "start": self.start,
            "delimiter": self.delimiter,
            "classify": self.classify,
        }
