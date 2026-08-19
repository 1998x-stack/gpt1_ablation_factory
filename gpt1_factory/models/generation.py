from __future__ import annotations

from typing import List, Optional

import torch

from .gpt_decoder import GPTDecoderLM


def default_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_from_scores(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample next-token indices from logits using temperature + top-k.

    Args:
        logits: (..., V) unnormalized logits.
        temperature: scale dividing logits (`>0`). Higher = more diverse.
        top_k: if `>0`, restrict sampling to the k highest-probability ids.
        rng: optional torch.Generator for reproducible sampling.

    Returns:
        (..., 1) long tensor of sampled ids.
    """
    temp = max(float(temperature), 1e-8)
    logits = logits / temp
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=rng)


@torch.no_grad()
def generate_model(
    model: GPTDecoderLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    device: Optional[str] = None,
    stop_ids: Optional[List[int]] = None,
) -> str:
    """Generate a continuation of ``prompt`` from an LM backbone.

    Args:
        model: a GPTDecoderLM (or LSTM baseline) exposing
            ``forward(input_ids=...) -> {'logits'}``.
        tokenizer: a `tokenizers.Tokenizer` with `.encode(str)` and
            `.decode(list[int])`.
        prompt: initial text to continue from.
        max_new_tokens: number of new tokens to sample before stopping.
        temperature: sampling temperature.
        top_k: top-k filtering (0 = disabled).
        device: device string; defaults to `default_device()`.
        stop_ids: token ids that end generation.

    Returns:
        The prompt plus generated tokens, decoded to a string.
    """
    device = device or default_device()
    model.to(device).eval()

    max_len = getattr(model, "max_len", None)
    if stop_ids is None:
        eos_id = tokenizer.token_to_id("</s>")
        stop_ids = [eos_id] if eos_id is not None else []

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(int(max_new_tokens)):
        # Do not exceed the model's positional-embedding budget.
        if max_len is not None and input_ids.size(1) >= max_len:
            break
        logits = model(input_ids=input_ids)["logits"][0, -1, :]
        next_id = int(sample_from_scores(logits, temperature, top_k).item())
        if next_id in stop_ids:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tokenizer.decode(input_ids[0].tolist())
