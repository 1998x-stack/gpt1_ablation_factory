# Local-Corpus Training + Text Generation — Design

**Date:** 2026-08-19
**Status:** Approved
**Goal:** Let the GPT-1 Ablation Factory train on a local raw-text corpus
(`sample_corpus.txt`, Grimm's Fairy Tales, ~104k words) with a small demo model,
and provide a text-generation CLI to "test with examples". Includes a
professional polish pass across code and docs.

## Context

* The repo is a factory-mode GPT-1 pretrain → finetune → ablation project with
  registry-driven datasets/models, YAML configs, BPE tokenization.
* It already reads local `.txt` corpora via `local_text_dir`, but only as a
  fallback after attempting network loads (`_load_books_like_split`). The local
  path is therefore not first-class and depends on optional hugingface hub
  access for some configs.
* No text-generation / inference tool exists; only pretrain/finetune/ablation
  CLIs.
* Local env: `/usr/bin/python3` (3.9.6) with torch 2.8, already has
  yaml/loguru/tensorboard/sklearn; `tokenizers`, `datasets`, `einops`,
  `transformers` were installed to run training + generation.

## Decisions

- **Scope:** small-scale demo model (4 layers, 256 hidden, 4 heads), trained
  on the local fairy-tales corpus for a few hundred steps, then generate
  continuations. Full 12-layer GPT-1 is out of scope for the demo but the
  project keeps its full configs.
- **"Test with examples" = text generation:** sample likely next tokens from
  the trained LM to continue prompts.

## Architecture / Components

1. **Local text dataset** — add a first-class `load_local_text` registered as
   `DATASETS["local_text"]` that reads `*.txt` from `local_text_dir` with the
   `datasets` `"text"` loader. No network dependency. Existing hub-based
   builders (`bookcorpusopen`, `glue`, `race`, `story_cloze`) stay unchanged.
   `pretrain.py` `_ensure_pretrain_data_defaults` keeps local_text_dir default.

2. **Vocab alignment fix** — after building the tokenizer, set the model's
   `vocab_size` to `tokenizer.get_vocab_size()` instead of trusting the yaml
   default (50257). Removes a latent embedding-index mismatch between trained
   BPE and `ModelConfig.vocab_size`.

3. **Generation module** — `gpt1_factory/models/generation.py`:
   - `sample(logits, temperature, top_k) -> Tensor`
   - `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k,
     stop_ids) -> str`
   CLI `gpt1_factory/cli/generate.py` loads a checkpoint + `bpe.json` from an
   existing run dir, instantiates the model (vocab from tokenizer), and
   generates continuations for prompts given via `--prompt` / `--input-file`.

4. **Small demo configs + scripts**:
   - `configs/model/gpt_mini.yaml` (4L / 256 / 4H, max_len 128)
   - `configs/data/books_local.yaml` (name `local_text`, vocab 4000,
     seq_len 128, batch_size 16)
   - `configs/pretrain_local.yaml` (warmup 150, ~600 steps, small saved
     interval)
   - `scripts/pretrain_local.sh`, `scripts/generate.sh`

5. **Polish pass** — normalize code docstrings to English (repo currently mixes
   EN/CN), tidy formatting, expand `README.md` with a documented
   "Train on your own `.txt` + generate" quickstart and a generated-samples
   section.

6. **Verification** — run a real short pretrain on `sample_corpus.txt`, run
   `generate` with fairy-tale prompts, run the existing + new tests, capture
   sample outputs into README/notes.

## Data flow

```
sample_corpus.txt  ->  data/text/  ->  DATASETS["local_text"]  ->  BPE.train
  -> TextDataset + LMTrainCollator  ->  PretrainTrainer  ->  checkpoint + bpe.json
                                   ->  cli/generate.py  ->  sampled continuation
```

## Error handling

- Generation raises a clear error if the checkpoint/tokenizer paths don't exist.
- `local_text` raises if `local_text_dir` is empty/missing.
- Sample loops safeguard against over-long repeats via stop-on-`</s>` and
  max_new_tokens cap.

## Testing

- Existing tests (`test_tokenizer`, `test_models`) must still pass.
- New: `tests/test_local_corpus.py` (local_text dataset loads a tiny txt
  fixture; vocab alignment) and `tests/test_generation.py` (top-k/temperature
  sampling shape; full generate returns a string with expected prompt prefix).