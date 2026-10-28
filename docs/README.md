# Documentation index

This directory separates **scientific contracts**, **active implementation plans**, and **historical design notes**.

| Document | Purpose | Status |
| --- | --- | --- |
| [PAPER_FIDELITY.md](PAPER_FIDELITY.md) | Paper-oriented model design, embedding/normalization choices, transfer fidelity and known deviations | Implemented contracts + open fidelity gaps |
| [TOKENIZER_TASK_PROTOCOL.md](TOKENIZER_TASK_PROTOCOL.md) | Frozen tokenizer artifacts, task special tokens, input serialization and checkpoint compatibility | Implemented contracts + historical differences |
| [PAPER_ABLATION_FACTORY.md](PAPER_ABLATION_FACTORY.md) | Section-by-section paper mapping, experimental matrices, reporting, evaluation and acceptance criteria | Active implementation / research roadmap |
| [Historical local-corpus plan](superpowers/plans/2026-08-19-local-corpus-training.md) | Original task breakdown for local text training | Historical reference |
| [Historical local-corpus design](superpowers/specs/2026-08-19-local-corpus-training-design.md) | Earlier design rationale and assumptions | Historical reference |

## Source of truth

- **Paper facts:** [Radford et al. (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), cited in research notes.
- **What currently runs:** `gpt1_factory/`, `configs/` and `tests/` on the checked-out branch.
- **How to run:** repository [README](../README.md).
- **How to contribute safely:** repository [AGENTS.md](../AGENTS.md).

These sources have different roles. A paper claim does not establish the code implements it; source code does not establish experiment performance; and a test pass does not establish scientific replication.

## Current milestones

1. **A — Experimental infrastructure:** configuration expansion, isolated runs, partial Table 5/transfer/λ matrix, logs, reports and contract tests.
2. **B — Dataset identity:** corpus licensing, manifests, version and split fingerprints, token sampling audit.
3. **C — Zero-shot fidelity:** RACE likelihood scorer exists; remaining published heuristics and time-indexed evaluations are incomplete.
4. **D — Empirical replication:** full budget, multiple seeds, paired uncertainty estimates, runtime/memory accounting, error analysis.

The root README intentionally summarizes this index rather than duplicating detailed paper experiments.
