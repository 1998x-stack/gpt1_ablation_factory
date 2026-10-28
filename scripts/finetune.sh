#!/usr/bin/env bash
set -euo pipefail
python -m gpt1_factory.cli.finetune --cfg configs/finetune_glue.yaml "$@"
