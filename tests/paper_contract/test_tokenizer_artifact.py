import json

import pytest

from gpt1_factory.data.text_bpe import BPEBuilder
from gpt1_factory.tokenization import (
    GPT1SpecialTokens,
    TokenizerArtifact,
    TokenizerCompatibilityError,
)


def test_tokenizer_artifact_roundtrip_and_special_ids(tmp_path):
    builder = BPEBuilder(tmp_path / "bpe", vocab_size=100, min_freq=1)
    tok = builder.train(["hello world", "another sample"])

    artifact = TokenizerArtifact.create(
        tok,
        tmp_path / "artifact",
    )
    loaded = TokenizerArtifact.load(tmp_path / "artifact")

    assert loaded.fingerprint == artifact.fingerprint
    assert loaded.vocab_size == tok.get_vocab_size()

    specials = GPT1SpecialTokens()
    assert loaded.special_token_ids["start"] == tok.token_to_id(specials.start)
    assert loaded.special_token_ids["delimiter"] == tok.token_to_id(
        specials.delimiter
    )
    assert loaded.special_token_ids["classify"] == tok.token_to_id(
        specials.classify
    )


def test_tokenizer_artifact_detects_file_tampering(tmp_path):
    builder = BPEBuilder(tmp_path / "bpe", vocab_size=100, min_freq=1)
    tok = builder.train(["hello world"])
    artifact = TokenizerArtifact.create(tok, tmp_path / "artifact")

    tokenizer_path = artifact.path / "tokenizer.json"
    tokenizer_path.write_text(
        tokenizer_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TokenizerCompatibilityError,
        match="fingerprint mismatch",
    ):
        TokenizerArtifact.load(artifact.path)


def test_manifest_is_human_inspectable(tmp_path):
    builder = BPEBuilder(tmp_path / "bpe", vocab_size=100, min_freq=1)
    tok = builder.train(["hello world"])
    artifact = TokenizerArtifact.create(tok, tmp_path / "artifact")

    manifest = json.loads(
        (artifact.path / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["format_version"] == 1
    assert manifest["fingerprint"] == artifact.fingerprint
    assert manifest["vocab_size"] == artifact.vocab_size
