# GPT-1 Paper Fidelity

This repository treats GPT-1 paper fidelity as an explicit, testable contract rather
than an informal naming convention.

## Architecture contract

The canonical `configs/model/gpt_small.yaml` now uses:

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

## Downstream correctness contract

The first fidelity pass also guarantees:

1. classification pooling uses an explicit classify position when supplied, or the
   last non-padding token rather than a padded final position;
2. multiple-choice tasks use one shared scalar score per candidate;
3. transfer-depth filtering really loads only the first K Transformer blocks;
4. fine-tuning gradient clipping includes task heads;
5. the best fine-tuning checkpoint includes task-head parameters;
6. fine-tuning learning rate warms up and then decays linearly.

## Intentionally deferred

Tokenizer artifact inheritance, exact GPT-1 special-token transforms, dataset
fingerprints, strict corpus resolution, and full paper experiment reproduction are
kept for follow-up PRs so this correctness change remains reviewable.
