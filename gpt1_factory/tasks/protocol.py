from __future__ import annotations

from dataclasses import dataclass

from tokenizers import Tokenizer

from ..tokenization import GPT1SpecialTokens, TokenizerCompatibilityError


@dataclass(frozen=True)
class TaskEncoding:
    """Single GPT-1 task traversal before batch padding."""

    input_ids: list[int]
    classify_position: int


class GPT1TaskProtocol:
    """Serialize structured downstream examples using GPT-1 task tokens."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_len: int,
        specials: GPT1SpecialTokens | None = None,
    ) -> None:
        if max_len < 3:
            raise ValueError("max_len must be at least 3 for GPT-1 task inputs.")

        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.specials = specials or GPT1SpecialTokens()

        self.start_id = self._required_id(self.specials.start)
        self.delimiter_id = self._required_id(self.specials.delimiter)
        self.classify_id = self._required_id(self.specials.classify)

    def encode_classification(self, text: str) -> TaskEncoding:
        body = self._encode_text(text)
        budget = self.max_len - 2
        ids = [self.start_id] + body[:budget] + [self.classify_id]
        return TaskEncoding(ids, len(ids) - 1)

    def encode_entailment(self, premise: str, hypothesis: str) -> TaskEncoding:
        return self._encode_pair(premise, hypothesis)

    def encode_similarity(
        self,
        first: str,
        second: str,
    ) -> tuple[TaskEncoding, TaskEncoding]:
        return (
            self._encode_pair(first, second),
            self._encode_pair(second, first),
        )

    def encode_multiple_choice(
        self,
        context: str,
        question: str | None,
        choice: str,
    ) -> TaskEncoding:
        prompt = context if not question else f"{context}\n{question}"
        return self._encode_pair(prompt, choice)

    def _encode_pair(self, first: str, second: str) -> TaskEncoding:
        left = self._encode_text(first)
        right = self._encode_text(second)
        left, right = _balanced_truncate(
            left,
            right,
            budget=self.max_len - 3,
        )
        ids = (
            [self.start_id]
            + left
            + [self.delimiter_id]
            + right
            + [self.classify_id]
        )
        return TaskEncoding(ids, len(ids) - 1)

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            str(text),
            add_special_tokens=False,
        ).ids

    def _required_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise TokenizerCompatibilityError(
                f"Tokenizer does not contain required GPT-1 task token {token!r}."
            )
        return int(token_id)


def _balanced_truncate(
    left: list[int],
    right: list[int],
    *,
    budget: int,
) -> tuple[list[int], list[int]]:
    if budget < 0:
        raise ValueError("budget must be non-negative")

    left = list(left)
    right = list(right)
    while len(left) + len(right) > budget:
        if len(left) >= len(right) and left:
            left.pop()
        elif right:
            right.pop()
        else:
            break
    return left, right
