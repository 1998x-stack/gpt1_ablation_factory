import pytest
import torch
import torch.nn as nn

from gpt1_factory.models.checkpoint import (
    load_pretrained_partial,
    save_checkpoint,
)
from gpt1_factory.models.gpt_decoder import GPTDecoderLM
from gpt1_factory.tokenization import TokenizerCompatibilityError


def test_checkpoint_persists_task_modules_and_metadata(tmp_path):
    backbone = nn.Linear(4, 4)
    head = nn.Linear(4, 2)
    path = tmp_path / "best.pt"

    save_checkpoint(
        path,
        backbone,
        step=7,
        extra_modules={"classification_head": head},
        metadata={"tokenizer_fingerprint": "abc123"},
    )

    payload = torch.load(path, map_location="cpu")

    assert payload["step"] == 7
    assert "classification_head" in payload["modules"]
    assert set(payload["modules"]["classification_head"]) == set(
        head.state_dict()
    )
    assert payload["metadata"]["tokenizer_fingerprint"] == "abc123"


def test_checkpoint_rejects_wrong_tokenizer_fingerprint(tmp_path):
    model = GPTDecoderLM(
        vocab_size=17,
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
    )
    path = tmp_path / "pretrained.pt"
    save_checkpoint(
        path,
        model,
        metadata={"tokenizer_fingerprint": "expected"},
    )

    target = GPTDecoderLM(
        vocab_size=17,
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
    )
    with pytest.raises(
        TokenizerCompatibilityError,
        match="Checkpoint/tokenizer mismatch",
    ):
        load_pretrained_partial(
            target,
            path,
            expected_tokenizer_fingerprint="different",
        )
