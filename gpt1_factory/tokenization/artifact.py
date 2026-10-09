from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from .special_tokens import GPT1SpecialTokens


class TokenizerCompatibilityError(RuntimeError):
    """Raised when a tokenizer artifact cannot be trusted for a checkpoint."""


@dataclass(frozen=True)
class TokenizerArtifact:
    """Immutable tokenizer payload persisted alongside a pretraining run."""

    path: Path
    tokenizer: Tokenizer
    manifest: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    @property
    def vocab_size(self) -> int:
        return int(self.manifest["vocab_size"])

    @property
    def special_token_ids(self) -> dict[str, int]:
        return {
            name: int(item["id"])
            for name, item in self.manifest["special_tokens"].items()
        }

    @classmethod
    def create(
        cls,
        tokenizer: Tokenizer,
        directory: str | Path,
        *,
        specials: GPT1SpecialTokens | None = None,
    ) -> "TokenizerArtifact":
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        specials = specials or GPT1SpecialTokens()

        special_tokens: dict[str, dict[str, Any]] = {}
        for name, token in specials.as_dict().items():
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise TokenizerCompatibilityError(
                    f"Tokenizer is missing required special token {token!r} ({name}). "
                    "Retrain the tokenizer with the GPT-1 special-token contract."
                )
            special_tokens[name] = {"token": token, "id": int(token_id)}

        tokenizer_path = directory / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        fingerprint = _sha256_file(tokenizer_path)

        manifest = {
            "format_version": 1,
            "tokenizer_file": tokenizer_path.name,
            "fingerprint": fingerprint,
            "vocab_size": int(tokenizer.get_vocab_size()),
            "special_tokens": special_tokens,
        }
        _write_json(directory / "manifest.json", manifest)
        return cls(path=directory, tokenizer=tokenizer, manifest=manifest)

    @classmethod
    def load(cls, path: str | Path) -> "TokenizerArtifact":
        path = Path(path)
        directory = path.parent if path.name == "manifest.json" else path
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Tokenizer artifact manifest not found: {manifest_path}. "
                "Run pretraining with the artifact-aware pipeline first."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise TokenizerCompatibilityError(
                f"Unsupported tokenizer artifact format: {manifest.get('format_version')!r}"
            )

        tokenizer_path = directory / str(manifest["tokenizer_file"])
        if not tokenizer_path.exists():
            raise TokenizerCompatibilityError(
                f"Tokenizer file referenced by manifest is missing: {tokenizer_path}"
            )

        actual_fingerprint = _sha256_file(tokenizer_path)
        expected_fingerprint = str(manifest["fingerprint"])
        if actual_fingerprint != expected_fingerprint:
            raise TokenizerCompatibilityError(
                "Tokenizer artifact fingerprint mismatch: "
                f"manifest={expected_fingerprint}, actual={actual_fingerprint}"
            )

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if int(tokenizer.get_vocab_size()) != int(manifest["vocab_size"]):
            raise TokenizerCompatibilityError(
                "Tokenizer vocab size does not match the artifact manifest."
            )

        for name, item in manifest["special_tokens"].items():
            actual_id = tokenizer.token_to_id(str(item["token"]))
            if actual_id != int(item["id"]):
                raise TokenizerCompatibilityError(
                    f"Special token mismatch for {name}: "
                    f"manifest={item['id']}, tokenizer={actual_id}"
                )

        return cls(path=directory, tokenizer=tokenizer, manifest=manifest)


def infer_tokenizer_artifact_path(checkpoint_path: str | Path) -> Path:
    """Infer <run>/tokenizer from <run>/checkpoints/<checkpoint>.pt."""

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name != "checkpoints":
        raise TokenizerCompatibilityError(
            "Cannot infer tokenizer artifact from checkpoint path "
            f"{checkpoint_path!s}; expected <run>/checkpoints/<file>.pt."
        )
    return checkpoint_path.parent.parent / "tokenizer"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
