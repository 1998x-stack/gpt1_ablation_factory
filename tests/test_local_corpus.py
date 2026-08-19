from __future__ import annotations

from pathlib import Path

from gpt1_factory.configs import DataConfig
from gpt1_factory.data.datasets import (
    load_local_text,
    DatasetBundle,
    resolve_model_vocab_size,
)
from gpt1_factory.data.text_bpe import BPEBuilder


def _make_config(text_dir, cache_dir, bpe_dir) -> DataConfig:
    return DataConfig(
        name="local_text",
        batch_size=2,
        num_workers=0,
        seq_len=32,
        local_text_dir=str(text_dir),
        cache_dir=str(cache_dir),
        bpe={"save_dir": str(bpe_dir), "vocab_size": 200, "min_freq": 1},
    )


def test_load_local_text(tmp_path: Path) -> None:
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "a.txt").write_text("Once upon a time there was a king who had three sons.")
    (text_dir / "b.txt").write_text("The princess slept in a tall tower for one hundred years.")

    bundle = load_local_text(_make_config(text_dir, tmp_path / "cache", tmp_path / "bpe"))

    assert isinstance(bundle, DatasetBundle)
    assert bundle.train is not None and bundle.valid is None and bundle.test is None
    assert bundle.tokenizer is not None
    # 200 vocab + 5 special tokens -> vocab_size between 200 and 205
    assert 200 <= bundle.tokenizer.get_vocab_size() <= 205
    ids = bundle.tokenizer.encode("Once upon a time").ids
    assert len(ids) > 0

    # BPE json is persisted so a second load reuses it
    second = load_local_text(text_config := _make_config(text_dir, tmp_path / "cache", tmp_path / "bpe"))
    assert second.tokenizer.get_vocab_size() == bundle.tokenizer.get_vocab_size()


def test_missing_dir_raises(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "does-not-exist", tmp_path / "cache", tmp_path / "bpe")
    try:
        load_local_text(cfg)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing local_text_dir")


def test_resolve_model_vocab_size() -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=100, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<sep>"])
    tok.train_from_iterator(["hello world", "second line"], trainer=trainer)

    assert resolve_model_vocab_size(tok, 50257) == tok.get_vocab_size()