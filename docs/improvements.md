# Portfolio improvement history

### 2026-11-05 09:19 UTC — ChatGPT Portfolio Iteration
- Run ID: 20261105T091924Z-a0fe83
- Repo / Base SHA: 1998x-stack/gpt1_ablation_factory / e7f0ff8a6825adb6e3aa98a18e5a325918cd430e
- Historical context: Default-branch docs/improvements.md returned GitHub 404 before changes. Reviewed README, recent commits and existing PRs (#1–#4 merged, no open PR). No prior portfolio history exists. Existing generation tests cover top-k, temperature, normal prompt, and stop IDs but not an empty prompt.
- Product problem & expected user value: A user running text generation with an empty CLI prompt receives an opaque tensor indexing failure instead of an actionable input-validation error.
- Engineering diagnosis & evidence: gpt1_factory/models/generation.py generate_model encodes the prompt and indexes logits[0, -1, :] without checking an empty token sequence. An isolated Torch reproduction produced IndexError: index -1 is out of bounds for dimension 1 with size 0.
- Options & decision: Empty-prompt validation (P2, benefit 4 + correctness 4 + evidence 5 + testability 4 - cost 1 - risk 1 = 15); temperature nonfinite handling (P2, 3+3+4+4-2-2=10); overlong prompt behavior (P2, 3+3+3+3-3-3=6). Selected the narrow input guard to preserve normal generation behavior.
- Changes: gpt1_factory/models/generation.py rejects empty tokenized input only when max_new_tokens > 0, with ValueError. tests/test_generation.py adds empty-prompt failure and zero-generation compatibility cases; no new dependencies or external API signatures.
- Validation: A (isolated reproduction only): executed a local Torch harness mirroring the relevant code path, verifying empty prompt raises ValueError, zero-generation empty prompt returns an empty string, and nonempty prompt retains normal output; 3/3 checks passed. B (baseline): pre-fix isolated Torch reproduction raised IndexError. D: the actual GitHub branch's pytest suite, Ruff, and integrated tokenizer/model checks were not run because the runtime has Torch but lacks tokenizers and cannot access GitHub from the shell. The isolated harness is not a substitute for those tests.
- Security & regression review: No secrets, external requests, permission changes, CI workflows, training data, model weights, or deployment changes. Normal nonempty prompts unchanged; potential compatibility change for callers relying on the prior empty-prompt IndexError.
- Risk & rollback: Low risk input validation; revert this branch's source and test commits if tokenizer-specific behavior is incompatible. Full pytest and CLI smoke test remain required before merge.
- Follow-up: Run full tests/test_generation.py with real tokenizers and a CLI smoke test on an empty prompt, then examine temperature NaN/zero handling as a separate issue.
- Commit: 14f8c8654f515741489963cab4c4a4b26fc0f532 (tests); 1ba597da0dcecf26b7acecb4fc4bca54b844ad70 (source); e91185e01e2403dd709f0422ff5cb5026a2d7bd8 (initial history)
- PR: https://github.com/1998x-stack/gpt1_ablation_factory/pull/5
