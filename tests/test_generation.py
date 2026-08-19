from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from gpt1_factory.models.generation import sample_from_scores, generate_model
from gpt1_factory.models.gpt_decoder import GPTDecoderLM
from gpt1_factory.models.heads import ClassificationHead


def _tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=40,
                         special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<sep>"])
    tok.train_from_iterator(["hello world", "how are you today",
                             "the quick brown fox", "jumps over the lazy dog"],
                            trainer=trainer)
    return tok


def test_sample_from_scores_respects_topk() -> None:
    logits = torch.randn(8, 100)
    sampled = sample_from_scores(logits, top_k=5)
    assert sampled.shape == (8, 1)
    top5 = torch.topk(logits, 5, dim=-1).indices
    for i in range(8):
        assert sampled[i, 0].item() in top5[i].tolist()


def test_sample_temperature_returns_same_dtype() -> None:
    logits = torch.randn(4, 50)
    out = sample_from_scores(logits, temperature=0.5, top_k=10, rng=torch.Generator().manual_seed(0))
    assert out.dtype == torch.long and out.shape == (4, 1)


def test_generate_model_returns_string_with_prompt() -> None:
    tok = _tiny_tokenizer()
    model = GPTDecoderLM(vocab_size=tok.get_vocab_size() + 20, n_layer=1, n_head=2,
                         d_model=16, d_ff=32, max_len=64)
    model.eval()
    text = generate_model(model, tok, "hello world", max_new_tokens=8, temperature=0.8,
                          top_k=10, device="cpu")
    # NOTE: the Whitespace-pretokenized BPE splits "hello" into subwords and decode
    # inserts spaces, so we assert against the *decoded* prompt rather than the raw
    # prompt string (the raw form can never appear verbatim under this tokenizer).
    prompt_ids = tok.encode("hello world", add_special_tokens=False).ids
    assert isinstance(text, str) and text.startswith(tok.decode(prompt_ids))