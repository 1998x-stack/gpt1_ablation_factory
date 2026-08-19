#!/usr/bin/env bash
set -e
/usr/bin/python3 -m gpt1_factory.cli.generate \
  --bpe runs/bpe_local/bpe.json \
  --ckpt runs/exp_local/checkpoints/latest.pt \
  --model-yaml configs/model/gpt_mini.yaml \
  --prompt "Once upon a time" \
  --max-new-tokens 80 --temperature 0.9 --top-k 60 --no-stop-on-eos \
  --num-samples 3
