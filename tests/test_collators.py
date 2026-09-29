import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

from gpt1_factory.data.collators import (
    ClassificationCollator,
    LMTrainCollator,
    MultiChoiceCollator,
    SimilarityCollator,
)
from gpt1_factory.tokenization import GPT1SpecialTokens


def _toy_tok():
    specials = GPT1SpecialTokens()
    tok = Tokenizer(BPE(unk_token=specials.unk))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(specials.ordered())
    tok.add_tokens(["hello", "world", "question", "choice", "other"])
    return tok


def test_lm_collator():
    tok = _toy_tok()
    coll = LMTrainCollator(tok, seq_len=8)
    batch = [{"text": "hello world"}]
    out = coll(batch)
    assert "input_ids" in out
    assert out["input_ids"].shape[1] == 8


def test_cls_collator_tracks_classify_token():
    tok = _toy_tok()
    coll = ClassificationCollator(
        tok,
        max_len=16,
        text_cols=("a", "b"),
        mode="entailment",
    )
    batch = [{"a": "hello", "b": "world", "label": 1}]
    out = coll(batch)

    pos = int(out["classify_positions"][0])
    classify_id = tok.token_to_id("_classify_")

    assert out["input_ids"].shape[1] == 16
    assert out["input_ids"][0, pos].item() == classify_id
    assert out["labels"].tolist() == [1]


def test_similarity_collator_emits_two_traversals():
    tok = _toy_tok()
    coll = SimilarityCollator(
        tok,
        max_len=12,
        text_cols=("a", "b"),
    )
    out = coll([{"a": "hello", "b": "world", "label": 1}])

    assert out["input_ids"].shape == (1, 2, 12)
    assert bool(out["similarity_pairs"].item())
    assert not torch.equal(
        out["input_ids"][0, 0],
        out["input_ids"][0, 1],
    )


def test_multiple_choice_label_extractor_does_not_require_label_field():
    tok = _toy_tok()
    coll = MultiChoiceCollator(
        tok,
        max_len=12,
        options_extractor=lambda ex: ex["options"],
        context_extractor=lambda ex: ex["context"],
        question_extractor=lambda ex: ex["question"],
        label_extractor=lambda ex: "AB".index(ex["answer"]),
    )
    out = coll(
        [
            {
                "context": "hello",
                "question": "question",
                "options": ["choice", "other"],
                "answer": "B",
            }
        ]
    )

    assert out["input_ids"].shape == (1, 2, 12)
    assert out["labels"].tolist() == [1]
    assert out["classify_positions"].shape == (1, 2)
