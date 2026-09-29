# Tokenizer Artifact and GPT-1 Task Protocol

This layer prevents a silent but severe transfer-learning failure: a downstream
dataset must never retrain a tokenizer whose integer ids have different meanings
from the ids used by the pretrained embedding matrix.

## Pretraining artifact

Every pretraining run now writes:

```text
<run>/
├── checkpoints/
│   ├── latest.pt
│   └── step_*.pt
└── tokenizer/
    ├── tokenizer.json
    └── manifest.json
```

The manifest records:

- artifact format version;
- vocabulary size;
- SHA-256 fingerprint of `tokenizer.json`;
- ids for pad, unknown, EOS, `_start_`, `_delimiter_`, and
  `_classify_`.

The same fingerprint is embedded in every pretraining checkpoint.

## Finetuning invariant

Finetuning loads the tokenizer artifact before constructing a dataset or model.

The pipeline then enforces all of the following:

1. downstream loaders receive the inherited tokenizer and are not allowed to train
   a new BPE;
2. the model vocabulary size is taken from the artifact;
3. the pretrained checkpoint must contain the same tokenizer fingerprint;
4. a missing or mismatched fingerprint fails before pretrained weights are loaded.

For random-initialization ablations, the model still uses the same tokenizer
artifact so the experiment changes pretraining, not token semantics.

## GPT-1 task tokens

The task protocol uses the OpenAI reference names:

```text
_start_
_delimiter_
_classify_
```

The repository reserves these tokens when training its BPE so their ids are stable
inside the artifact. This differs slightly from the original TensorFlow release,
which appended task tokens for finetuning; the serialization semantics are the
same, while stable reservation makes tokenizer/checkpoint identity directly
verifiable.

## Task serialization

Classification:

```text
_start_ text _classify_
```

Entailment:

```text
_start_ premise _delimiter_ hypothesis _classify_
```

Similarity/paraphrase:

```text
_start_ A _delimiter_ B _classify_
_start_ B _delimiter_ A _classify_
```

Both similarity traversals pass through the shared backbone. Their
`_classify_` representations are summed before the classification/regression
head.

Multiple choice:

```text
_start_ context/question _delimiter_ choice_i _classify_
```

Every candidate receives one shared scalar score and candidates are compared along
the choice dimension.

## Truncation contract

Task serialization truncates text content before assembly. Structural tokens are
never truncated, so `_classify_` always remains present and its exact position is
returned by the collator.

## Migration from older runs

A tokenizer created before this contract will not contain all required task tokens.
The loader intentionally rejects it instead of mutating token ids silently.

Use a fresh BPE directory or remove the stale tokenizer and rerun pretraining. New
checkpoints and their tokenizer artifact must be generated together.
