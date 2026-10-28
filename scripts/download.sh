#!/usr/bin/env bash
set -euo pipefail
# Downloads/caches public datasets; this is not an exact BooksCorpus guarantee.
python -m gpt1_factory.cli.download --target all "$@"
