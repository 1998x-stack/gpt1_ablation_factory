from torch.utils.data import DataLoader

from gpt1_factory.configs import ExpConfig, FinetuneConfig
from gpt1_factory.data.collators import SimilarityCollator
from gpt1_factory.data.text_bpe import BPEBuilder
from gpt1_factory.models.gpt_decoder import GPTDecoderLM
from gpt1_factory.trainers.finetune_trainer import FinetuneTrainer


def test_similarity_trainer_runs_dual_traversal_end_to_end(tmp_path):
    tok = BPEBuilder(
        tmp_path / "bpe",
        vocab_size=100,
        min_freq=1,
    ).train(
        [
            "alpha beta",
            "beta alpha",
            "same pair",
            "different pair",
        ]
    )
    collator = SimilarityCollator(
        tok,
        max_len=12,
        text_cols=("a", "b"),
        return_lm_labels=False,
    )

    records = [
        {"a": "alpha beta", "b": "beta alpha", "label": 1},
        {"a": "alpha beta", "b": "different pair", "label": 0},
    ]
    loader = DataLoader(
        records,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
    )

    model = GPTDecoderLM(
        vocab_size=tok.get_vocab_size(),
        n_layer=1,
        n_head=1,
        d_model=8,
        d_ff=16,
        max_len=12,
        dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
    )
    trainer = FinetuneTrainer(
        ExpConfig(out_dir=str(tmp_path / "run")),
        FinetuneConfig(
            pretrained_path="",
            aux_lm_lambda=0.0,
            epochs=1,
            lr=1e-3,
            amp=False,
        ),
        model,
        num_labels=2,
        train_loader=loader,
        valid_loader=loader,
        task_name="mrpc",
        tokenizer_fingerprint="test-fingerprint",
    )

    best, detail = trainer.train()

    assert isinstance(best, float)
    assert "acc" in detail
    assert "f1" in detail
    assert (tmp_path / "run/checkpoints/best.pt").exists()
