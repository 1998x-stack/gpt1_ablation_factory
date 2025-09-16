#!/usr/bin/env bash
set -e
# Download everything into gpt1_ablation_factory/data/
python -m gpt1_factory.cli.download --target all
