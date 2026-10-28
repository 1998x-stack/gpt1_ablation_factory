# Repository instructions for coding agents

These instructions apply to the entire repository. Read this file before editing code, tests, experiments, or documentation.

## Mission and evidence standard

Develop a **reproducible, paper-guided GPT-1 ablation laboratory**. Preserve the existing PyTorch factory-mode architecture; improve scientific validity and operational reliability before adding breadth.

Canonical primary source:
- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training*: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

Repository evidence:
- `docs/PAPER_FIDELITY.md` — architecture and fidelity boundary.
- `docs/TOKENIZER_TASK_PROTOCOL.md` — token IDs, artifacts and task formatting.
- `docs/PAPER_ABLATION_FACTORY.md` — research matrix, output contract and outstanding gaps.
- `docs/README.md` — document map and implementation stages.

**Do not** equate source-code implementation, unit-test success, or a dry-run plan with reproduced paper results. Separate "paper-reported", "implemented but not measured", "measured here", and "planned".

## Working rules

1. Start by inspecting the relevant source, configuration, tests, and paper evidence. Describe the experimental hypothesis before making nontrivial algorithm changes.
2. Prefer a focused branch and an inspectable PR. Do not merge into `main` or delete historical work without explicit authorization.
3. Keep pure configuration, factories, models, datasets, training loops, metrics, and reporting as separate concerns. Add a new variant by configuration/registration rather than copying a training pipeline.
4. Limit the patch to the requested behavior. Avoid broad renames or moving legacy research files unless imports, references and tests are updated together.
5. Never write results or checkpoints into source folders. Generated data belongs under ignored dataset paths; runs and logs belong under `runs/`.
6. When a design or implementation intentionally differs from the paper, annotate the deviation and add an ablation or verification contract where appropriate.
7. Do not use unlicensed text or fabricate corpora, checkpoints, metrics, graphics or benchmark wins. Do not leak evaluation-set labels into training.
8. Keep README, command examples, config files, tests and `docs/` synchronized whenever CLI or default paths change.

## Architecture map

| Concern | Primary files |
| --- | --- |
| Registration and configs | `gpt1_factory/registry.py`, `gpt1_factory/configs.py`, `configs/` |
| Backbone and head contracts | `gpt1_factory/models/` |
| Data source, tokenization and batching | `gpt1_factory/data/`, `gpt1_factory/tokenization/` |
| Input serialization and task metrics | `gpt1_factory/tasks/` |
| Pre-training and fine-tuning | `gpt1_factory/trainers/` |
| Experiment expansion / statistical report | `gpt1_factory/experiments/` |
| CLI entrypoints | `gpt1_factory/cli/` |
| Paper-contract and regression tests | `tests/paper_contract/`, `tests/` |
| Short shell wrappers | `scripts/` |

Run commands from the repository root using the current environment's `python`, **not** hardcoded `/usr/bin/python3`. Python 3.10+; install `requirements.txt` before editable installation, and install `pytest` for tests.

## Non-negotiable technical contracts

### Model and optimization

- `gpt_small.yaml` is the paper-oriented 12L/768D/12H/3072 Post-LN model; `gpt_mini.yaml` is a 4-layer local validation profile. Don't silently change canonical defaults.
- Attention must obey causal masking; masked positions must not affect logits or gradients. Honor each model's `max_len`.
- The auxiliary objective has the form `L = L_task + lambda * L_LM`. Log its components separately and verify padding positions use `ignore_index=-100`.
- Control partial transfer by actual checkpoint tensor keys and record which tensors are transferred. The current K=0 transfer path retains **non-block** parameters and must **not** be described as a pure embedding-only transfer without changing the loader and its tests.
- LSTM controls require an **LSTM-trained** checkpoint, not Transformer weights. Preserve token-ID mapping and document parameter/compute-budget differences.
- Clip unscaled gradients with AMP. Reject nonfinite losses, shape mismatches and missing checkpoint/tokenizer inputs explicitly.

### Data and tokenizer

- Reuse the pretraining tokenizer artifact and validate its SHA-256 fingerprint when transferring.
- Never retrain a task-local tokenizer during fine-tuning. Keep special-token IDs (`_start_`, `_delimiter_`, `_classify_`) consistent with the artifact.
- For NLI, similarity and multiple-choice tasks, honor `tasks/protocol.py` and the correct pooled classification position. Symmetric similarity must evaluate both input orders.
- Do not silently substitute BookCorpusOpen with unrelated corpora in paper-oriented runs. If an explicitly permitted fallback is used for a demo, mark the result as **non-paper-equivalent**.
- Fix the dataset, split, sampling protocol, preprocessing, vocabulary, maximum length and seed within each controlled comparison.

### Experiment identity and output

- Study axes belong in the YAML matrix, not hardcoded conditional chains in trainers.
- Each arm × task × seed × transfer depth × auxiliary weight receives an isolated directory and resolved config, with a stable config hash and provenance.
- Do not overwrite a different run's artifacts. Re-running a completed arm may be skipped only when its configuration identity matches.
- `--resume` currently means **skip completed runs**; it does not implement exact restart from optimizer/scheduler/RNG state. Never claim otherwise.
- For each run, preserve `resolved.yaml`, `provenance.json`, `runner.log`, `events.jsonl`, `metrics.json`, `checkpoints/` and `tb/` as appropriate.
- Use Loguru for diagnostic events; structured JSONL for step/epoch metrics; TensorBoard for time-series visualization. Graphs are derived from real artifacts, not copied or simulated values.

### Metrics and figures

- Report task-specific primary metrics. Accuracy, F1, Pearson, Spearman and Matthews are not directly interchangeable scales.
- Evaluate on validation data for model selection. Maintain correct GLUE task schemas (MNLI: premise/hypothesis; QNLI: question/sentence; RTE: sentence1/sentence2; MRPC/QQP: symmetric).
- Compare matched seeds and tasks. If no matched control exists, show it as unavailable instead of inventing a delta.
- Standard chart types: learning/learning-rate curves, per-task ablation comparisons, transferred-depth curves, and checkpoint-indexed zero-shot trajectories.
- State sample size, dispersion, missing seeds, model and data limitations. No "state-of-the-art", "reproduced" or statistical significance claims without measurements and an appropriate test.

## Safe commands and acceptance checks

```bash
# First check the independent configs and plan. The command below does NOT train.
python -m gpt1_factory.cli.ablation_factory --cfg configs/ablations.yaml
python -m gpt1_factory.cli.ablation_factory --cfg configs/paper_ablations.yaml

# Run regression and contract tests after code/config changes.
python -m pytest -q
python -m pytest -q tests/paper_contract tests/test_experiment_factory.py

# Full training requires separately prepared inputs; NEVER trigger it as a quick test.
python -m gpt1_factory.cli.ablation_factory --cfg configs/ablations.yaml --execute

# Reports only summarize existing, completed experiment artifacts.
python -m gpt1_factory.cli.report_ablations --runs runs/paper_ablations --out runs/ablation_report
```

For documentation-only edits: verify every relative link and command against the current tree, and confirm scientific status language. For changes to models/collators/checkpoints: add a targeted regression test before full-suite tests. For changes to experimental matrices: assert run counts, unique output directories, unchanged control factors and parameter/schema compatibility.

## Changes requiring additional review

- Data-license or corpus-identity changes.
- Tokenizer/special-token IDs, checkpoint compatibility and transfer semantics.
- Evaluation split, label mapping, primary metric or sample-selection changes.
- Any change that makes a comparison unfair through different sequence length, tokenization, compute budget or accidental configuration inheritance.
- Publishing model weights, generated corpora or empirical benchmark claims.

## Definition of done

A change is ready for review only when:

- Implementation matches the documented hypothesis and advertised CLI.
- Paper-facing assumptions and limitations are explicit and cited in docs.
- The config resolves correctly, names and paths are valid, and no run collision is introduced.
- Tests cover new behavior, failure modes and preservation of existing paper-contract invariants.
- Formatting, README, docs, scripts and any experiment matrix remain internally consistent.
- The summary states exact validation performed, what could not be exercised locally, and any remaining research limitations.
