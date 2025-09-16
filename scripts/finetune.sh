#!/usr/bin/env bash
set -e
python -m gpt1_factory.cli.finetune --cfg configs/finetune_glue.yaml
