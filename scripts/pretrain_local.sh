#!/usr/bin/env bash
set -euo pipefail
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml "$@"
