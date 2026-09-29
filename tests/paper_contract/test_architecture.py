import torch
import torch.nn as nn

from gpt1_factory.models.gpt_decoder import Block, GPTDecoderLM


def test_paper_defaults_use_post_ln_without_final_norm():
    model = GPTDecoderLM(
        vocab_size=101,
        n_layer=2,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
    )

    assert all(block.norm_style == "post_ln" for block in model.blocks)
    assert isinstance(model.ln_f, nn.Identity)
    assert model.final_layer_norm is False


def test_post_ln_block_matches_reference_equations():
    torch.manual_seed(0)
    block = Block(
        d_model=8,
        n_head=1,
        d_ff=16,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        max_len=4,
        norm_style="post_ln",
    ).eval()

    x = torch.randn(2, 4, 8)
    actual = block(x)

    n = block.ln1(x + block.attn(x))
    expected = block.ln2(n + block.mlp(n))

    torch.testing.assert_close(actual, expected)


def test_language_model_projection_is_weight_tied_by_default():
    model = GPTDecoderLM(
        vocab_size=101,
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=8,
    )

    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
