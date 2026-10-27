"""Track zero-shot RACE accuracy over a series of LM pretraining checkpoints.

This is a paper-inspired heuristic, not a claim of exact historical tokenization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..experiments.factory import load_config
from ..experiments.zero_shot import score_choices
from ..models.checkpoint import load_pretrained_partial
from ..registry import MODELS
from ..tokenization import TokenizerArtifact
from ..models import GPTDecoderLM, LSTMBaseline  # ensure factories register


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="List of step_*.pt and/or latest.pt checkpoint files")
    parser.add_argument("--model-config", default="configs/model/gpt_small.yaml")
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--out", default="runs/zero_shot_race/metrics.json")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    from datasets import load_dataset

    artifact = TokenizerArtifact.load(args.tokenizer_artifact)
    model_cfg = load_config(args.model_config)["model"]
    model_cfg["vocab_size"] = artifact.vocab_size
    max_len = model_cfg.get("max_len", 512)
    raw = load_dataset("race", "all", split="validation")
    rows = []
    for ckpt in args.checkpoints:
        model = MODELS.create(model_cfg["name"], **{
            k: v for k, v in model_cfg.items() if k != "name"
        })
        load_pretrained_partial(model, ckpt,
                                expected_tokenizer_fingerprint=artifact.fingerprint)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        total = correct = 0
        for ex in raw.select(range(min(len(raw), args.limit))):
            prompt = str(ex["article"]) + "\n" + str(ex["question"])
            scores = score_choices(model, artifact.tokenizer, prompt,
                                   [str(x) for x in ex["options"]], max_len)
            prediction = max(range(len(scores)), key=lambda k: scores[k])
            correct += int(prediction == "ABCD".index(ex["answer"]))
            total += 1
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        rows.append({"checkpoint": str(ckpt), "pretrain_step": int(state.get("step", 0)),
                     "task": "RACE", "samples": total, "accuracy": correct / total,
                     "heuristic": "mean_completion_logprob"})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
