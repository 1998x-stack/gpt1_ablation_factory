from __future__ import annotations

import argparse

import torch
import yaml
from loguru import logger
from tokenizers import Tokenizer

from ..configs import ModelConfig, dataclass_from_dict
from ..models.generation import default_device, generate_model
from ..registry import MODELS


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_model(model_yaml: str, bpe_path: str, ckpt_path: str):
    """Instantiate a model whose vocab matches the trained BPE, then load weights."""
    model_cfg = dataclass_from_dict(ModelConfig, _load_yaml(model_yaml).get("model", {}))
    tokenizer = Tokenizer.from_file(bpe_path)
    model_cfg.vocab_size = tokenizer.get_vocab_size()
    model_kwargs = {k: v for k, v in model_cfg.__dict__.items() if k != "name"}
    model = MODELS.create(model_cfg.name, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained LM.")
    parser.add_argument("--model-yaml", type=str, default="configs/model/gpt_mini.yaml")
    parser.add_argument("--bpe", type=str, default="runs/bpe_local/bpe.json",
                        help="Path to a trained BPE tokenizer (bpe.json)")
    parser.add_argument("--ckpt", type=str, default="runs/exp_local/checkpoints/latest.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--no-stop-on-eos", action="store_true",
                        help="Do not stop generation on the </s> token (line/para boundary).")
    args = parser.parse_args()

    model, tokenizer = _load_model(args.model_yaml, args.bpe, args.ckpt)
    logger.info(f"Loaded model from {args.ckpt} on {default_device()}")

    stop_ids = [] if args.no_stop_on_eos else None
    for i in range(1, args.num_samples + 1):
        text = generate_model(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_ids=stop_ids,
        )
        print(f"--- sample {i} ---")
        print(text)


if __name__ == "__main__":
    main()
