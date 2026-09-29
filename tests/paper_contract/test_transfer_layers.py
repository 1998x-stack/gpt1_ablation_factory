import torch

from gpt1_factory.models.checkpoint import (
    load_pretrained_partial,
    select_pretrained_state_dict,
)
from gpt1_factory.models.gpt_decoder import GPTDecoderLM


def test_partial_transfer_keeps_only_first_k_blocks():
    state = {
        "tok_emb.weight": torch.tensor([1.0]),
        "pos_emb.weight": torch.tensor([2.0]),
        "blocks.0.attn.qkv.weight": torch.tensor([3.0]),
        "blocks.1.attn.qkv.weight": torch.tensor([4.0]),
        "blocks.2.attn.qkv.weight": torch.tensor([5.0]),
        "ln_f.weight": torch.tensor([6.0]),
    }

    selected = select_pretrained_state_dict(state, transfer_layers=2)

    assert "tok_emb.weight" in selected
    assert "pos_emb.weight" in selected
    assert "blocks.0.attn.qkv.weight" in selected
    assert "blocks.1.attn.qkv.weight" in selected
    assert "blocks.2.attn.qkv.weight" not in selected
    assert "ln_f.weight" in selected


def test_zero_layer_transfer_keeps_shared_non_block_weights():
    state = {
        "tok_emb.weight": torch.tensor([1.0]),
        "blocks.0.attn.qkv.weight": torch.tensor([2.0]),
    }

    selected = select_pretrained_state_dict(state, transfer_layers=0)

    assert list(selected) == ["tok_emb.weight"]


def test_full_transfer_returns_all_weights():
    state = {
        "tok_emb.weight": torch.tensor([1.0]),
        "blocks.11.mlp.0.weight": torch.tensor([2.0]),
    }

    assert set(select_pretrained_state_dict(state, -1)) == set(state)


def test_legacy_untied_checkpoint_does_not_overwrite_tied_embeddings(tmp_path):
    source = GPTDecoderLM(
        vocab_size=17,
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
        tie_emb=False,
    )
    with torch.no_grad():
        source.tok_emb.weight.fill_(1.0)
        source.lm_head.weight.fill_(2.0)

    path = tmp_path / "legacy.pt"
    torch.save({"model": source.state_dict(), "step": 0}, path)

    target = GPTDecoderLM(
        vocab_size=17,
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
        tie_emb=True,
    )
    load_pretrained_partial(target, path)

    torch.testing.assert_close(
        target.tok_emb.weight,
        torch.ones_like(target.tok_emb.weight),
    )
    assert target.tok_emb.weight.data_ptr() == target.lm_head.weight.data_ptr()
