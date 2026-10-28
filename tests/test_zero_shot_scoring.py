import pytest
import torch

from gpt1_factory.experiments.zero_shot import continuation_mean_logprob


class FixedLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, input_ids):
        b, t = input_ids.shape
        logits = torch.zeros((b, t, 8), device=input_ids.device)
        logits[:, :, 2] = 2.0
        return {"logits": logits}


def test_candidate_scoring_uses_completion_only():
    model = FixedLM()
    hit = continuation_mean_logprob(model, [4, 5, 6], [2], 5)
    miss = continuation_mean_logprob(model, [4, 5, 6], [3], 5)
    assert hit > miss


def test_truncation_keeps_exact_answer_likelihood():
    model = FixedLM()
    a = continuation_mean_logprob(model, [4] * 30, [2, 2], 5)
    b = continuation_mean_logprob(model, [4] * 2, [2, 2], 5)
    assert a == pytest.approx(b)


def test_long_completion_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        continuation_mean_logprob(FixedLM(), [1, 2], [2] * 5, 5)
