# GPT-1 Paper Fidelity

This repository treats GPT-1 paper fidelity as an explicit, testable contract rather
than an informal naming convention.

## Architecture contract

The canonical `configs/model/gpt_small.yaml` uses:

- 12 Transformer blocks
- hidden size 768
- 12 attention heads
- feed-forward size 3072
- learned positional embeddings
- GELU
- Post-LayerNorm residual blocks
- no extra final LayerNorm after the last block
- tied token-embedding / language-model output weights
- normal initialization with standard deviation 0.02

The previous repository architecture remains available as
`configs/model/gpt_modern_preln.yaml` so architectural changes can be ablated
instead of being hidden implementation details.

## Transfer and downstream contract

The fidelity layer now guarantees:

1. pretraining emits an immutable tokenizer artifact and embeds its SHA-256
   fingerprint in checkpoints;
2. downstream datasets inherit that tokenizer instead of training task-local BPEs;
3. checkpoint loading rejects a missing or different tokenizer fingerprint;
4. GPT-1 task inputs use explicit `_start_`, `_delimiter_`, and
   `_classify_` tokens;
5. classification and entailment pool the explicit `_classify_` position;
6. semantic-similarity tasks run both A->B and B->A traversals and sum their
   representations before the head;
7. multiple-choice tasks use one shared scalar score per candidate;
8. transfer-depth filtering loads only the first K Transformer blocks;
9. fine-tuning gradient clipping includes task heads;
10. the best fine-tuning checkpoint includes task-head parameters;
11. fine-tuning learning rate warms up and then decays linearly.

See `docs/TOKENIZER_TASK_PROTOCOL.md` for the artifact and serialization details.

## Known fidelity boundary

The task-token serialization mirrors the OpenAI reference implementation, but this
repository reserves task tokens during BPE construction so their ids can be part of
an immutable pretraining artifact. The original TensorFlow release appended three
task tokens at finetuning time. A future ablation can isolate that initialization
difference if exact historical token-add timing is required.

## Intentionally deferred

Dataset fingerprints, strict BookCorpus resolution, complete resume/RNG state,
paper Table 5 and Figure 2 experiment suites, multi-seed aggregation, and zero-shot
training trajectories remain follow-up work.
