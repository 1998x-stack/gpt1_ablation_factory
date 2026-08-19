from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from gpt1_factory.models.generation import sample_from_scores, generate_model
from gpt1_factory.models.gpt_decoder import GPTDecoderLM


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


class _AlwaysEosModel(torch.nn.Module):
    """Dummy LM that overwhelmingly samples `</s>`; records the longest sequence seen."""

    def __init__(self, eos_id: int, vocab: int) -> None:
        super().__init__()
        self.eos_id = eos_id
        self.vocab = vocab
        self.max_len = 128
        self.longest = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        self.longest = max(self.longest, input_ids.size(1))
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab)
        logits[:, -1, self.eos_id] = 10.0
        return {"logits": logits}


def test_empty_stop_ids_disables_eos_stop() -> None:
    tok = _tiny_tokenizer()
    eos = tok.token_to_id("</s>")
    vocab = tok.get_vocab_size() + 20

    stop_default = _AlwaysEosModel(eos, vocab)
    no_stop = _AlwaysEosModel(eos, vocab)

    generate_model(stop_default, tok, "hello world", max_new_tokens=8, device="cpu")
    generate_model(no_stop, tok, "hello world", max_new_tokens=8, stop_ids=[], device="cpu")

    prompt_len = len(tok.encode("hello world", add_special_tokens=False).ids)
    # Default stops almost immediately on </s>; passing an empty list ignores it
    # and runs the full budget (prompt + max_new_tokens, capped at max_len).
    # The counter sees forward-call lengths, so the last appended token (the
    # max_new_tokens-th) hasn't been fed through by the time generation ends.
    assert stop_default.longest < no_stop.longest
    assert no_stop.longest >= prompt_len + 8 - 1
