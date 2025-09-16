import torch
from gpt1_factory.data.collators import LMTrainCollator, ClassificationCollator
from tokenizers import Tokenizer
from tokenizers.models import BPE

def _toy_tok():
    tok = Tokenizer(BPE())
    tok.add_special_tokens(["<pad>", "<s>", "</s>", "<sep>", "<unk>"])
    tok.add_tokens(["hello", "world"])
    return tok

def test_lm_collator():
    tok = _toy_tok()
    coll = LMTrainCollator(tok, seq_len=8)
    batch = [{"text": "hello world"}]
    out = coll(batch)
    assert "input_ids" in out and out["input_ids"].shape[1] == 8

def test_cls_collator():
    tok = _toy_tok()
    coll = ClassificationCollator(tok, max_len=16, text_cols=("a", "b"))
    batch = [{"a": "hello", "b": "world", "label": 1}]
    out = coll(batch)
    assert out["input_ids"].shape[1] == 16
    assert "labels" in out
