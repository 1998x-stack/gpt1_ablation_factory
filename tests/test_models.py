import torch
from gpt1_factory.models.gpt_decoder import GPTDecoderLM
from gpt1_factory.models.lstm_baseline import LSTMBaseline

def test_gpt_forward():
    m = GPTDecoderLM(vocab_size=1000, n_layer=2, n_head=2, d_model=64, d_ff=128, max_len=16)
    x = torch.randint(0, 1000, (2, 16))
    out = m(x, labels=x)
    assert "loss" in out and out["loss"] is not None

def test_lstm_forward():
    m = LSTMBaseline(vocab_size=1000, d_model=64, lstm_hidden=128, max_len=16)
    x = torch.randint(0, 1000, (2, 16))
    out = m(x, labels=x)
    assert "loss" in out and out["loss"] is not None
