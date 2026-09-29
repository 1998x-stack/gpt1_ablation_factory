import torch

from gpt1_factory.models.heads import (
    ChoiceScoringHead,
    pool_sequence_state,
)


def test_pool_sequence_state_uses_last_non_padding_position():
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]],
            [[4.0, 4.0], [5.0, 5.0], [8.0, 8.0], [7.0, 7.0]],
        ]
    )
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ]
    )

    pooled = pool_sequence_state(hidden, attention_mask=mask)

    torch.testing.assert_close(
        pooled,
        torch.tensor([[3.0, 3.0], [5.0, 5.0]]),
    )


def test_explicit_classify_positions_take_priority():
    hidden = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.long)

    pooled = pool_sequence_state(
        hidden,
        attention_mask=mask,
        classify_positions=torch.tensor([1, 2]),
    )

    torch.testing.assert_close(
        pooled,
        torch.stack([hidden[0, 1], hidden[1, 2]]),
    )


def test_choice_head_returns_one_score_per_candidate():
    head = ChoiceScoringHead(d_model=8, dropout=0.0)
    hidden = torch.randn(3, 4, 8)

    scores = head(hidden)

    assert scores.shape == (3, 4)
