#!/usr/bin/env bash
set -e
python -m gpt1_factory.cli.run_ablation --cfg configs/ablations.yaml
