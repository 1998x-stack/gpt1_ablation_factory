"""Causal LM candidate scoring, implementing the GPT-1 §5 RACE heuristic."""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def continuation_mean_logprob(model, prefix_ids: list[int],
                              completion_ids: list[int], max_len: int) -> float:
    """Mean log p(answer tokens | context), excluding context tokens and padding.

    For token index i, logits at position i-1 predict token i. Retain at
    least one context token after truncation. Never score context tokens.
    """
    if not prefix_ids or not completion_ids:
        raise ValueError("Both prefix and completion must contain tokens")
    if len(completion_ids) >= max_len:
        raise ValueError("Completion does not fit with at least one context token")
    prefix = prefix_ids[-(max_len - len(completion_ids)):]
    sequence = prefix + completion_ids
    input_ids = torch.tensor([sequence[:-1]], dtype=torch.long,
                             device=next(model.parameters()).device)
    logits = model(input_ids=input_ids)["logits"][0]
    logp = F.log_softmax(logits, dim=-1)
    # First completion token follows the last token of the prefix.
    positions = torch.arange(len(prefix) - 1, len(sequence) - 1,
                             device=input_ids.device)
    targets = torch.tensor(completion_ids, dtype=torch.long, device=input_ids.device)
    return float(logp[positions, targets].mean().item())


def score_choices(model, tokenizer, prompt: str, choices: list[str],
                  max_len: int) -> list[float]:
    """Zero-shot scores for RACE-style choices (no downstream classifier)."""
    prefix = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not choices:
        raise ValueError("At least one completion choice is required")
    scores = []
    for candidate in choices:
        answer = tokenizer.encode(" " + candidate, add_special_tokens=False).ids
        scores.append(continuation_mean_logprob(model, prefix, answer, max_len))
    return scores
