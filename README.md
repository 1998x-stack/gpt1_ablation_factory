# GPT-1 Ablation Factory

**A paper-guided, factory-mode PyTorch research workspace for generative pre-training, supervised transfer, and controlled ablation studies.**

**Reference:** Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018) · [Original paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

> **Research status:** The architecture, experiment factory, logging, reporting, and contract tests are implemented. The repository does **not** claim to have reproduced the paper's benchmark scores. Full-scale pre-training, exact corpus equivalence, and statistical validation remain separate research tasks.

## Overview

The project connects three stages:

1. **Pre-train** a decoder-only Transformer with causal language modeling, preserving its tokenizer artifact and checkpoint metadata.
2. **Transfer** pretrained weights to downstream NLI, sentence classification, sentence-pair similarity, and multiple-choice tasks.
3. **Ablate** one research variable at a time with explicit run identity, independent outputs, and task-specific evaluation.

The **paper-oriented default** is 12 Transformer layers, 768 hidden dimensions, 12 attention heads, 3,072 feed-forward dimensions, learned positional embeddings, causal attention, GELU, Post-LayerNorm blocks, and tied token/output embeddings. A small 4-layer configuration is provided for local pipeline validation. A modern Pre-LN alternative and an LSTM baseline remain available for comparisons.

### Capabilities and boundaries

| Area | Available implementation | Important boundary |
| --- | --- | --- |
| Modeling | GPT decoder, mini/small profiles, modern Pre-LN comparison, LSTM | Exact historical initialization and training equivalence require further verification |
| Pre-training | Causal LM, local-text/BookCorpusOpen loaders, checkpoint and tokenizer artifact | Alternative corpora are **not** interchangeable with original BooksCorpus |
| Transfer | Full/partial loading, frozen tokenizer mapping, task-specific heads, auxiliary LM objective | No end-to-end published benchmark reproduction yet |
| Task protocols | GLUE subsets, RACE and Story Cloze loaders; explicit classification tokens and symmetric similarity | Complete paper task coverage is not implemented |
| Experiment factory | YAML suites × variants × tasks × seeds × transfer depth × LM coefficient | Plans are configurations, not already-run experiments |
| Observability | Loguru, TensorBoard, JSONL events, checkpoint metadata, validation metrics | Dataset fingerprints, exact interrupted-run resume and full resource accounting are pending |
| Reporting | Task-specific plots, offline HTML, CSV/JSON summaries, matched-seed differences | Bootstrap confidence intervals and full statistical inference are pending |
| Zero-shot | Checkpoint-wise, candidate-likelihood RACE evaluation | This is a paper-inspired partial evaluation, not all Figure 2 heuristics |

## Repository layout

```text
configs/
  model/                 GPT-1, mini, modern Pre-LN, LSTM model definitions
  data/                  Dataset and tokenization settings
  pretrain_local.yaml    Small local training exercise
  pretrain_books.yaml    Paper-scale Transformer pre-training entry
  pretrain_lstm.yaml     Independently pre-trained LSTM control
  finetune_glue.yaml     Downstream fine-tuning configuration
  finetune_smoke.yaml    Mini-model/128-token smoke configuration
  ablations.yaml         Three-arm smoke matrix
  paper_ablations.yaml   162-run paper-inspired research plan

gpt1_factory/
  cli/                   Python -m command entry points
  models/                Decoder, LSTM, heads, generation, checkpoints
  data/                  Dataset adapters, collators, BPE builder
  tokenization/          Tokenizer fingerprint/artifact contract
  tasks/                 Task serialization, metrics and evaluation
  trainers/              Pre-training, fine-tuning, zero-shot routines
  experiments/           Validated job factory, scoring, reports and plots
  utils/                 Logging, seeds, schedules and TensorBoard

tests/
  paper_contract/        Paper-facing invariant tests
  test_experiment_factory.py
  test_zero_shot_scoring.py
  ...                    Component/integration-oriented tests

docs/
  README.md              Documentation map and project status
  PAPER_FIDELITY.md      Architecture fidelity and known deviations
  TOKENIZER_TASK_PROTOCOL.md
  PAPER_ABLATION_FACTORY.md
  superpowers/           Archived implementation plans and specifications

scripts/                 Thin shell wrappers around Python CLI entry points
AGENTS.md                Repository-specific guidance for coding agents
```

## Installation

Python **3.10+** is required. Run all commands from the **repository root**.

```bash
git clone https://github.com/1998x-stack/gpt1_ablation_factory.git
cd gpt1_ablation_factory

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

This repository currently keeps its runtime library set in `requirements.txt`. Training requires PyTorch; CUDA hardware is strongly recommended for paper-scale experiments. Installing dependencies does not download training datasets.

## Quick start: local mini-model

The bundled `sample_corpus.txt` is a **pipeline demo**, not an official GPT-1 reproduction dataset.

```bash
# 1. Supply local training text under the configured (git-ignored) directory.
mkdir -p gpt1_ablation_factory/data/text
cp sample_corpus.txt gpt1_ablation_factory/data/text/sample_corpus.txt

# 2. Train the 4-layer mini model and produce a checkpoint/tokenizer artifact.
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml

# 3. Generate a continuation from the resulting checkpoint.
python -m gpt1_factory.cli.generate \
  --model-yaml configs/model/gpt_mini.yaml \
  --bpe runs/bpe_local/bpe.json \
  --ckpt runs/exp_local/checkpoints/latest.pt \
  --prompt "Once upon a time" --max-new-tokens 64

# 4. Print the three-run SST-2 smoke plan (no training).
python -m gpt1_factory.cli.ablation_factory --cfg configs/ablations.yaml
```

After the mini checkpoint and tokenizer artifact exist, execute the smoke experiments:

```bash
python -m gpt1_factory.cli.ablation_factory \
  --cfg configs/ablations.yaml --execute --resume --fail-fast
```

This requires downloading the SST-2 task data. The smoke configuration is defined in `configs/finetune_smoke.yaml`. It sets `data.max_len: 128` to match the mini model's context window and uses independent run directories.

## Paper-inspired experiment matrix

`configs/paper_ablations.yaml` expands into **162 planned fine-tuning runs** (not 162 pretrained models).

| Suite | Arms | Tasks | Seeds | Planned runs |
| --- | --- | --- | --- | ---: |
| `paper_table5` | Full, no pretraining, no auxiliary LM, pretrained LSTM | 8 GLUE tasks | 3 | 96 |
| `paper_transfer` | First K transferred blocks (K = 0, 3, 6, 9, 12) | MNLI, QNLI | 3 | 30 |
| `lambda_sweep` | Auxiliary LM weight (0, 0.1, 0.5, 1.0) | SST-2, MNLI, MRPC | 3 | 36 |
| **Total** | | | | **162** |

Run the plan without side effects:

```bash
python -m gpt1_factory.cli.ablation_factory --cfg configs/paper_ablations.yaml
```

**Before execution:** prepare the Transformer checkpoint and its tokenizer artifact, then train the LSTM using the inherited tokenizer. Paper-scale training is compute-intensive and strict BookCorpusOpen loading fails rather than silently substituting a different dataset.

```bash
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_books.yaml
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_lstm.yaml

python -m gpt1_factory.cli.ablation_factory \
  --cfg configs/paper_ablations.yaml --execute --resume --fail-fast
```

**Caution:** The original paper's corpus, BPE details, pre-training optimization, transfer protocol, and task evaluations are not guaranteed to match this implementation. In particular, the K = 0 transfer loader also includes non-block parameters: it is not a mathematically strict "embeddings only" intervention. Record such deviations in experimental interpretations. See [paper-to-code mapping](docs/PAPER_ABLATION_FACTORY.md).

## Results and visualization

```bash
# Consolidate completed experiments; no scores are fabricated for missing runs.
python -m gpt1_factory.cli.report_ablations \
  --runs runs/paper_ablations --out runs/ablation_report

# Inspect per-step training curves.
tensorboard --logdir runs/paper_ablations
```

Each executed run lives in `runs/paper_ablations/<unique-run-id>/`:

| Artifact | Meaning |
| --- | --- |
| `resolved.yaml` | Fully resolved experiment configuration |
| `provenance.json` | Run identity, config hash, paths and status |
| `runner.log` | Captured child process output |
| `events.jsonl` | Train/validation metrics stream |
| `metrics.json` | Best validation epoch and primary task metric |
| `checkpoints/best.pt` | Saved backbone and downstream heads |
| `tb/` | TensorBoard scalar events |

Reporting creates `runs/ablation_report/dashboard.html`, `summary.json`, CSV tables and per-task PNG charts. Compare results within tasks before drawing conclusions; accuracy, F1, Matthews correlation, and Pearson correlation have different semantics. The `--resume` switch skips fully completed matching runs; it does **not** restore optimizer and random state after an interrupted step.

## Zero-shot checkpoint evaluation

An initial RACE candidate-likelihood scorer is available:

```bash
python -m gpt1_factory.cli.zero_shot_race \
  --model-config configs/model/gpt_small.yaml \
  --tokenizer-artifact runs/exp_pretrain_books/tokenizer \
  --checkpoints runs/exp_pretrain_books/checkpoints/step_10000.pt \
  --limit 250 --out runs/zero_shot_race/metrics.json
```

The checkpoint path above is illustrative and must exist. The resulting accuracy is **not** presented as reproduction of the complete zero-shot Figure 2.

## Reproducibility and contribution

Research runs should document the paper section and hypothesis, the sole manipulated factor, dataset and tokenizer identities, model/optimizer settings, random seeds, and the task-specific primary metric. Preserve baseline behavior unless there is an explicit change request. Never interpret missing metrics as zero or report synthetic results as measured evidence.

For design contracts and implementation details, start with [documentation index](docs/README.md). For coding assistants and review conventions, read [AGENTS.md](AGENTS.md). The research factory and repository documentation are available on the `main` branch via [PR #3](https://github.com/1998x-stack/gpt1_ablation_factory/pull/3) and [PR #4](https://github.com/1998x-stack/gpt1_ablation_factory/pull/4). Published GPT-1 benchmark replication remains unverified.
