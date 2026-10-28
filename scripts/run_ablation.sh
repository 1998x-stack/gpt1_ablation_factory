#!/usr/bin/env bash
set -euo pipefail
# Defaults to a dry-run; pass --execute to launch the smoke experiments.
python -m gpt1_factory.cli.ablation_factory --cfg configs/ablations.yaml "$@"
