# Local-Corpus Training + Text Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the GPT-1 Ablation Factory train on a local raw-text corpus (`sample_corpus.txt`) with a small demo model, and generate text continuations from the trained model, while polishing code and docs.

**Architecture:** Add a first-class `local_text` dataset builder (reads `data/text/*.txt`, no network), align model `vocab_size` to the trained BPE tokenizer, add a sampler + `generate` module and a `cli/generate.py`, add small demo configs/scripts, run a real short pretrain + generation to verify, then polish docstrings and README.

**Tech Stack:** Python 3.9 (`/usr/bin/python3`), PyTorch 2.8, `tokenizers` 0.22, `datasets` 4.5, `einops`, `yaml`, `loguru`. All deps already installed. Run all commands with `/usr/bin/python3` (there is no `python` on PATH).

## Global Constraints

- Use `/usr/bin/python3` for ALL runtime invocations (plain `python` is not on PATH).
- Every test must pass with: `/usr/bin/python3 -m pytest <test> -v`
- Do NOT change the existing hub-based dataset builders (`bookcorpusopen`, `glue`, `race`, `story_cloze`) or the finetune/ablation CLI behavior.
- Small demo model: 4 layers, 256 hidden, 4 heads, `max_len` 128, BPE vocab 4000.
- `sample_corpus.txt` is already copied to `gpt1_ablation_factory/data/text/sample_corpus.txt`.
- Commit after every task with the exact message given.

---

### Task 1: Local-text dataset builder + tests

**Files:**
- Modify: `gpt1_factory/data/datasets.py`
- Test: `tests/test_local_corpus.py` (create)

**Interfaces:**
- Produces:
  - `load_local_text(cfg: DataConfig) -> DatasetBundle` registered as `DATASETS["local_text"]`.
  - `resolve_model_vocab_size(tokenizer, configured: int) -> int` — returns the trained tokenizer's vocab size (used by Task 2 and 4).
  - Reuses `_text_iter`, `BPEBuilder`, `LMTrainCollator`, `DatasetBundle`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_local_corpus.py`:

```python
from __future__ import annotations

from pathlib import Path

from gpt1_factory.configs import DataConfig
from gpt1_factory.data.datasets import (
    load_local_text,
    DatasetBundle,
    resolve_model_vocab_size,
)
from gpt1_factory.data.text_bpe import BPEBuilder


def _make_config(text_dir, cache_dir, bpe_dir) -> DataConfig:
    return DataConfig(
        name="local_text",
        batch_size=2,
        num_workers=0,
        seq_len=32,
        local_text_dir=str(text_dir),
        cache_dir=str(cache_dir),
        bpe={"save_dir": str(bpe_dir), "vocab_size": 200, "min_freq": 1},
    )


def test_load_local_text(tmp_path: Path) -> None:
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "a.txt").write_text("Once upon a time there was a king who had three sons.")
    (text_dir / "b.txt").write_text("The princess slept in a tall tower for one hundred years.")

    bundle = load_local_text(_make_config(text_dir, tmp_path / "cache", tmp_path / "bpe"))

    assert isinstance(bundle, DatasetBundle)
    assert bundle.train is not None and bundle.valid is None and bundle.test is None
    assert bundle.tokenizer is not None
    # 200 vocab + 5 special tokens -> vocab_size between 200 and 205
    assert 200 <= bundle.tokenizer.get_vocab_size() <= 205
    ids = bundle.tokenizer.encode("Once upon a time").ids
    assert len(ids) > 0

    # BPE json is persisted so a second load reuses it
    second = load_local_text(text_config := _make_config(text_dir, tmp_path / "cache", tmp_path / "bpe"))
    assert second.tokenizer.get_vocab_size() == bundle.tokenizer.get_vocab_size()


def test_missing_dir_raises(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path / "does-not-exist", tmp_path / "cache", tmp_path / "bpe")
    try:
        load_local_text(cfg)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for missing local_text_dir")


def test_resolve_model_vocab_size() -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=100, special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<sep>"])
    tok.train_from_iterator(["hello world", "second line"], trainer=trainer)

    assert resolve_model_vocab_size(tok, 50257) == tok.get_vocab_size()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_local_corpus.py -q`
Expected: FAIL (no `load_local_text`, `resolve_model_vocab_size`, no local_text dir behavior).

- [ ] **Step 3: Implement**

In `gpt1_factory/data/datasets.py`, add after `_load_books_like_split`:

```python
def resolve_model_vocab_size(tokenizer: Any, configured: int) -> int:
    """Prefer the trained tokenizer's vocabulary size over the config default.

    The model embedding/output heads must exactly match the BPE vocab the
    corpus was tokenized with, otherwise input indices can exceed the
    embedding table.
    """
    return tokenizer.get_vocab_size()


@DATASETS.register("local_text")
def load_local_text(cfg: DataConfig) -> DatasetBundle:
    """Load a directory of raw ``*.txt`` files as a pretraining corpus.

    This is the first-class, network-free path for training on your own
    text files (e.g. ``data/text/*.txt``). It reuses the same BPE + LM
    collator pipeline as the book-corpus loader.
    """
    text_dir = Path(cfg.local_text_dir or "gpt1_ablation_factory/data/text")
    if not text_dir.exists():
        raise FileNotFoundError(f"local_text corpus dir not found: {text_dir}")
    files = sorted(glob(str(text_dir / "**/*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *.txt files found in local_text_dir: {text_dir}")

    raw = datasets.load_dataset("text", data_files={"train": files},
                                cache_dir=cfg.cache_dir)["train"]

    bpe_cfg = cfg.bpe or {}
    save_dir = bpe_cfg.get("save_dir", "runs/bpe_local")
    builder = BPEBuilder(
        save_dir,
        bpe_cfg.get("vocab_size", 4000),
        bpe_cfg.get("min_freq", 2),
    )
    tok = builder.load_or_train(_text_iter(raw, None))
    collator = LMTrainCollator(tok, seq_len=cfg.seq_len or 256)
    return DatasetBundle(train=raw, valid=None, test=None,
                         tokenizer=tok, collator=collator)
```

Note: `datasets.load_dataset("text", ...)` returns a lazy dataset; calling
`bundle.train` inside `_text_iter` iterates it, which is what feeds BPE
training. This matches the existing `bookcorpusopen` pattern.

- [ ] **Step 4: Run test to verify it passes**

Run: `/usr/bin/python3 -m pytest tests/test_local_corpus.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_local_corpus.py gpt1_factory/data/datasets.py
git commit -m "feat(data): add local_text dataset builder + vocab resolution"
```

---

### Task 2: Align pretrain model vocab to trained tokenizer

**Files:**
- Modify: `gpt1_factory/cli/pretrain.py`
- Test: covered by Task 1 `test_resolve_model_vocab_size` + integration in Task 5.

**Interfaces:**
- Consumes: `load_dataset_factory(cfg)` -> `bundle.tokenizer`; `resolve_model_vocab_size(tokenizer, configured)`.
- Produces: a model whose `vocab_size` equals the trained BPE vocab.

- [ ] **Step 1: Implement the alignment**

In `gpt1_factory/cli/pretrain.py`, import `resolve_model_vocab_size` and, right after building the bundle, force the model config vocab to the tokenizer's size:

```python
from ..data import load_dataset_factory, resolve_model_vocab_size
# ...inside main(), replace the model-build block:
    bundle = load_dataset_factory(data_cfg)
    # Align embedding/output vocab with the trained BPE tokenizer.
    model_cfg.vocab_size = resolve_model_vocab_size(bundle.tokenizer, model_cfg.vocab_size)
    train_loader = DataLoader(...)
```

The existing lines that build `train_loader` and `model` stay, but the
`model_cfg.vocab_size` line must precede `MODELS.create`.

- [ ] **Step 2: Sanity check the CLI imports cleanly**

Run: `/usr/bin/python3 -c "import gpt1_factory.cli.pretrain" && echo OK`
Expected: `OK`, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add gpt1_factory/cli/pretrain.py
git commit -m "fix(pretrain): align model vocab_size with trained tokenizer"
```

---

### Task 3: Generation module + tests

**Files:**
- Create: `gpt1_factory/models/generation.py`
- Test: `tests/test_generation.py` (create)

**Interfaces:**
- Produces:
  - `sample_from_scores(logits, temperature=1.0, top_k=0, rng=None) -> torch.Tensor`
  - `generate(model, tokenizer, prompt, max_new_tokens=64, temperature=1.0,
    top_k=0, device=None, stop_ids=None) -> str`
  - `resolve_generator(device) -> str` (cuda if available else cpu)

- [ ] **Step 1: Write the failing test**

Create `tests/test_generation.py`:

```python
from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from gpt1_factory.models.generation import sample_from_scores, generate_model
from gpt1_factory.models.gpt_decoder import GPTDecoderLM
from gpt1_factory.models.heads import ClassificationHead


def _tiny_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=40,
                         special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<sep>"])
    tok.train_from_iterator(["hello world", "how are you today",
                             "the quick brown fox", "jumps over the lazy dog"],
                            trainer=trainer)
    return tok


def test_sample_from_scores_respects_topk() -> None:
    logits = torch.randn(8, 100)
    sampled = sample_from_scores(logits, top_k=5)
    assert sampled.shape == (8, 1)
    top5 = torch.topk(logits, 5, dim=-1).indices
    for i in range(8):
        assert sampled[i, 0].item() in top5[i].tolist()


def test_sample_temperature_returns_same_dtype() -> None:
    logits = torch.randn(4, 50)
    out = sample_from_scores(logits, temperature=0.5, top_k=10, rng=torch.Generator().manual_seed(0))
    assert out.dtype == torch.long and out.shape == (4, 1)


def test_generate_model_returns_string_with_prompt() -> None:
    tok = _tiny_tokenizer()
    model = GPTDecoderLM(vocab_size=tok.get_vocab_size() + 20, n_layer=1, n_head=2,
                         d_model=16, d_ff=32, max_len=64)
    model.eval()
    text = generate_model(model, tok, "hello world", max_new_tokens=8, temperature=0.8,
                          top_k=10, device="cpu")
    assert isinstance(text, str) and text.startswith("hello world")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/usr/bin/python3 -m pytest tests/test_generation.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named
'gpt1_factory.models.generation'`).

- [ ] **Step 3: Implement**

Create `gpt1_factory/models/generation.py`:

```python
from __future__ import annotations

from typing import List, Optional

import torch

from .gpt_decoder import GPTDecoderLM


def default_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_from_scores(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample next-token indices from logits using temperature + top-k.

    Args:
        logits: (..., V) unnormalized logits.
        temperature: scale dividing logits (`>0`). Higher = more diverse.
        top_k: if `>0`, restrict sampling to the k highest-probability ids.
        rng: optional torch.Generator for reproducible sampling.

    Returns:
        (..., 1) long tensor of sampled ids.
    """
    temp = max(float(temperature), 1e-8)
    logits = logits / temp
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=rng)


@torch.no_grad()
def generate_model(
    model: GPTDecoderLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    device: Optional[str] = None,
    stop_ids: Optional[List[int]] = None,
) -> str:
    """Generate a continuation of ``prompt`` from an LM backbone.

    Args:
        model: a GPTDecoderLM (or LSTM baseline) exposing
            ``forward(input_ids=...) -> {'logits'}``.
        tokenizer: a `tokenizers.Tokenizer` with `.encode(str)` and
            `.decode(list[int])`.
        prompt: initial text to continue from.
        max_new_tokens: number of new tokens to sample before stopping.
        temperature: sampling temperature.
        top_k: top-k filtering (0 = disabled).
        device: device string; defaults to `default_generator()`.
        stop_ids: token ids that end generation.

    Returns:
        The prompt plus generated tokens, decoded to a string.
    """
    device = device or default_device()
    model.to(device).eval()

    eos_id = tokenizer.token_to_id("</s>")
    stop_ids = stop_ids or ([eos_id] if eos_id is not None else [])

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(int(max_new_tokens)):
        logits = model(input_ids=input_ids)["logits"][0, -1, :]
        next_id = int(sample_from_scores(logits, temperature, top_k).item())
        if next_id in stop_ids:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tokenizer.decode(input_ids[0].tolist())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/usr/bin/python3 -m pytest tests/test_generation.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_generation.py gpt1_factory/models/generation.py
git commit -m "feat(generation): add temperature/top-k sampler and generate"
```

---

### Task 4: `generate` CLI

**Files:**
- Create: `gpt1_factory/cli/generate.py`

**Interfaces:**
- Consumes: `registry.MODELS`, `dataclass_from_dict`, `tokenizers.Tokenizer.from_file`,
  `generate_model`, `default_device`.
- Produces: a `main()` callable runnable as
  `/usr/bin/python3 -m gpt1_factory.cli.generate --bpe <bpe.json> --ckpt <latest.pt> --prompt "..."`.

- [ ] **Step 1: Implement the CLI**

Create `gpt1_factory/cli/generate.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from loguru import logger
from tokenizers import Tokenizer

from ..configs import ModelConfig, dataclass_from_dict
from ..models.generation import default_device, generate_model
from ..registry import MODELS


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_model(model_yaml: str, bpe_path: str, ckpt_path: str):
    """Instantiate a model whose vocab matches the trained BPE, then load weights."""
    model_cfg = dataclass_from_dict(ModelConfig, _load_yaml(model_yaml).get("model", {}))
    tokenizer = Tokenizer.from_file(bpe_path)
    model_cfg.vocab_size = tokenizer.get_vocab_size()
    model_kwargs = {k: v for k, v in model_cfg.__dict__.items() if k != "name"}
    model = MODELS.create(model_cfg.name, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained LM.")
    parser.add_argument("--model-yaml", type=str, default="configs/model/gpt_mini.yaml")
    parser.add_argument("--bpe", type=str, default="runs/bpe_local/bpe.json",
                        help="Path to a trained BPE tokenizer (bpe.json)")
    parser.add_argument("--ckpt", type=str, default="runs/exp_local/checkpoints/latest.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = _load_model(args.model_yaml, args.bpe, args.ckpt)
    logger.info(f"Loaded model from {args.ckpt} on {default_device()}")

    for i in range(1, args.num_samples + 1):
        text = generate_model(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"--- sample {i} ---")
        print(text)


if __name__ == "__main__":
    main()
```

This file wires the pieces from Task 3 with concrete paths: the checkpoint
gives model weights, the BPE file gives the tokenizer, and `generate_model`
runs autoregressive sampling.

- [ ] **Step 2: Import sanity check**

Run: `/usr/bin/python3 -c "import gpt1_factory.cli.generate" && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add gpt1_factory/cli/generate.py
git commit -m "feat(cli): add generate CLI for trained checkpoints"
```

---

### Task 5: Demo configs + scripts

**Files:**
- Create: `configs/model/gpt_mini.yaml`, `configs/data/books_local.yaml`,
  `configs/pretrain_local.yaml`, `scripts/pretrain_local.sh`, `scripts/generate.sh`

**Interfaces:**
- Consumes: `local_text` dataset, `gpt_decoder` model from Tasks 1–4.
- Produces: runnable configs (e.g. `configs/pretrain_local.yaml`) and shell wrappers.

- [ ] **Step 1: Create model config**

Create `configs/model/gpt_mini.yaml`:

```yaml
model:
  name: gpt_decoder
  vocab_size: 2000          # overridden by trained tokenizer
  n_layer: 4
  n_head: 4
  d_model: 256
  d_ff: 1024
  max_len: 128
  dropout: 0.1
  attn_dropout: 0.1
  resid_dropout: 0.1
  layer_norm_eps: 1.0e-5
  tie_emb: false
  gelu: true
```

- [ ] **Step 2: Create data config**

Create `configs/data/books_local.yaml`:

```yaml
data:
  name: local_text
  text_column: "text"
  seq_len: 128
  batch_size: 16
  num_workers: 2
  local_text_dir: "gpt1_ablation_factory/data/text"
  bpe:
    train: true
    vocab_size: 4000
    min_freq: 2
    save_dir: "runs/bpe_local"
```

- [ ] **Step 3: Create pretrain config**

Create `configs/pretrain_local.yaml`:

```yaml
exp:
  out_dir: "runs/exp_local"
  seed: 42

include:
  - "configs/model/gpt_mini.yaml"
  - "configs/data/books_local.yaml"

optim:
  lr: 0.0003
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps: 150
  max_steps: 1000
  scheduler: "cosine"
  grad_clip: 1.0
  amp: false

checkpoint:
  save_every: 100
  keep_last: 3
```

- [ ] **Step 4: Create shell scripts**

Create `scripts/pretrain_local.sh`:

```bash
#!/usr/bin/env bash
set -e
/usr/bin/python3 -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml
```

Create `scripts/generate.sh`:

```bash
#!/usr/bin/env bash
set -e
/usr/bin/python3 -m gpt1_factory.cli.generate \
  --bpe runs/bpe_local/bpe.json \
  --ckpt runs/exp_local/checkpoints/latest.pt \
  --model-yaml configs/model/gpt_mini.yaml \
  --prompt "Once upon a time" \
  --num-samples 3
```

Make both executable: `chmod +x scripts/pretrain_local.sh scripts/generate.sh`

- [ ] **Step 5: Smoke-check config is parseable**

Run: `/usr/bin/python3 -c "
import yaml
for p in ['configs/model/gpt_mini.yaml','configs/data/books_local.yaml','configs/pretrain_local.yaml']:
    yaml.safe_load(open(p)); print('ok', p)
"`
Expected: all three print `ok`.

- [ ] **Step 6: Commit**

```bash
git add configs/model/gpt_mini.yaml configs/data/books_local.yaml \
        configs/pretrain_local.yaml scripts/pretrain_local.sh scripts/generate.sh
git commit -m "feat: small demo configs + pretrain/generate scripts"
```

---

### Task 6: End-to-end verification (train + generate on sample_corpus.txt)

**Files:**
- Run-time only (no new source files). May create `runs/exp_local/...` artifacts (git-ignored).

**Interfaces:**
- Consumes: `configs/pretrain_local.yaml`, `sample_corpus.txt` at
  `gpt1_ablation_factory/data/text/`.

- [ ] **Step 1: Train the small model**

Run: `cd <repo> && /usr/bin/python3 -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml`
Expected: logs `[pretrain] step=...` appear; a `runs/exp_local/checkpoints/latest.pt`
and `runs/exp_local/log.txt` are produced. On a CPU this may take ~5-15 minutes;
run with `timeout` if needed, but must reach at least ~150 steps to produce a
checkpoint if playing with `save_every`.

Note: if step count is < `save_every` the latest-only checkpoint logic still
writes `latest.pt` on each save; confirm `runs/exp_local/checkpoints/latest.pt`
exists before proceeding.

- [ ] **Step 2: Generate continuations from the model**

Run: `/usr/bin/python3 -m gpt1_factory.cli.generate \
  --bpe runs/bpe_local/bpe.json --ckpt runs/exp_local/checkpoints/latest.pt \
  --prompt "Once upon a time" --num-samples 3`
Expected: prints 3 text samples beginning with the prompt.

- [ ] **Step 3: Run the full test suite**

Run: `/usr/bin/python3 -m pytest tests/ -q`
Expected: all tests pass (existing tokenizer/models/collators tests plus the new
local_corpus and generation tests).

- [ ] **Step 4: Commit run artifacts note (no source change)**

If the generated samples are worth keeping, add them to `README.md` in Task 7
(polish); otherwise leave runs/ untracked.

---

### Task 7: Polish pass — docstrings + README

**Files:**
- Modify: `README.md`, `gpt1_factory/models/gpt_decoder.py`,
  `gpt1_factory/models/lstm_baseline.py`, `gpt1_factory/models/heads.py`,
  `gpt1_factory/data/collators.py`, `gpt1_factory/registry.py`,
  `gpt1_factory/configs.py`, `gpt1_factory/trainers/*.py`,
  `gpt1_factory/utils/*.py`, `gpt1_factory/tasks/*.py`
- No behavior changes; only comments/docstrings/README.

**Interfaces:**
- Preserves all existing public signatures and return contracts.

- [ ] **Step 1: Normalize docstrings to English**

For each file above, convert Chinese docstring bodies/comments to concise
English, keeping function signatures, parameters and behavior unchanged. Do not
rename functions or change defaults. Keep class docstrings as one-sentence
English purpose statements.

- [ ] **Step 2: Rewrite the README local-demo section**

Add a "Local corpus quickstart (train + generate)" section that documents:

```
/usr/bin/python3 -m pip install -r requirements.txt   # if needed
cp sample_corpus.txt gpt1_ablation_factory/data/text/
/usr/bin/python3 -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml
/usr/bin/python3 -m gpt1_factory.cli.generate --bpe runs/bpe_local/bpe.json \
  --ckpt runs/exp_local/checkpoints/latest.pt --prompt "Once upon a time"
```

Plus a short paragraph noting the small model overfits a ~100k-word corpus and
is a demo, not a production GPT-1.

- [ ] **Step 3: Copy verified generation samples**

Paste 2-3 `generate` outputs from Task 6 (the real samples) into the README
under a "Sample continuations" block.

- [ ] **Step 4: Run the full test suite again**

Run: `/usr/bin/python3 -m pytest tests/ -q`
Expected: pass (docstring edits must not break imports).

- [ ] **Step 5: Commit**

```bash
git add README.md gpt1_factory/models gpt1_factory/data/collators.py \
        gpt1_factory/registry.py gpt1_factory/configs.py gpt1_factory/trainers \
        gpt1_factory/utils gpt1_factory/tasks
git commit -m "chore: professionalize docstrings and README with local demo guides"
```

---

## Self-Review

- **Spec coverage:** Spec items 1-6 map to Tasks 1,2 (local_text+vocab), 3-4
  (generation+CLI), 5 (configs/scripts), 6 (verify), 7 (polish/README). All
  covered.
- **Consistency:** `resolve_model_vocab_size` used in Tasks 1/2;
  `generate_model`/`default_device`/`sample_from_scores` used in Tasks 3/4 with
  identical names throughout.
- **Placeholders:** The only human-in-the-loop gap is running the real training
  in Task 6, which is captured as explicit steps with expected artifacts.