# GPT1 Ablation Factory

A pluggable, factory-mode project to reproduce GPT-1 style pretraining → finetuning → ablation:
- Decoder-only Transformer (12L, 768H, 12 heads, FFN 3072) with GELU, learned pos-emb.
- LSTM baseline (single-layer 2048) for ablation.
- Auxiliary LM loss during finetuning (λ=0.5) switchable.
- Transfer layers control: load first K layers from the pretrained checkpoint for finetuning ablation.
- Input formatting for NLI / QA / Paraphrase / Classification tasks (GLUE, RACE, StoryCloze).
- Factory registries for datasets, models, trainers; YAML-configured experiments; Loguru + TensorBoard.

```
gpt1_ablation_factory/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ setup.cfg
├─ configs/
│  ├─ pretrain_books.yaml
│  ├─ pretrain_local.yaml
│  ├─ finetune_glue.yaml
│  ├─ ablations.yaml
│  ├─ model/
│  │  ├─ gpt_mini.yaml
│  │  ├─ gpt_small.yaml
│  │  └─ lstm_baseline.yaml
│  └─ data/
│     ├─ books_corpus_open.yaml
│     ├─ books_local.yaml
│     ├─ glue_mnli.yaml
│     ├─ glue_sst2.yaml
│     ├─ race.yaml
│     └─ story_cloze.yaml
├─ scripts/
│  ├─ pretrain.sh
│  ├─ pretrain_local.sh
│  ├─ generate.sh
│  ├─ finetune.sh
│  ├─ download.sh
│  └─ run_ablation.sh
├─ gpt1_factory/
│  ├─ __init__.py
│  ├─ configs.py
│  ├─ registry.py
│  ├─ utils/
│  │  ├─ logging.py
│  │  ├─ seed.py
│  │  ├─ distributed.py
│  │  ├─ tensorboard.py
│  │  └─ schedules.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ datasets.py
│  │  ├─ collators.py
│  │  └─ text_bpe.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ gpt_decoder.py
│  │  ├─ lstm_baseline.py
│  │  ├─ heads.py
│  │  ├─ generation.py
│  │  └─ checkpoint.py
│  ├─ tasks/
│  │  ├─ __init__.py
│  │  ├─ formatting.py
│  │  ├─ glue_taskmap.py
│  │  └─ metrics.py
│  ├─ trainers/
│  │  ├─ __init__.py
│  │  ├─ pretrain_trainer.py
│  │  ├─ finetune_trainer.py
│  │  └─ zero_shot.py
│  └─ cli/
│     ├─ pretrain.py
│     ├─ generate.py
│     ├─ finetune.py
│     ├─ download.py
│     └─ run_ablation.py
└─ tests/
   ├─ test_tokenizer.py
   ├─ test_models.py
   ├─ test_collators.py
   ├─ test_generation.py
   └─ test_local_corpus.py
```

## Quickstart

```bash
# 1) Install
/usr/bin/python3 -m venv .venv && source .venv/bin/activate
/usr/bin/python3 -m pip install -U pip
/usr/bin/python3 -m pip install -e .

# 2) Pretrain (BooksCorpusOpen; BPE trained on-the-fly)
/usr/bin/python3 -m gpt1_factory.cli.pretrain --cfg configs/pretrain_books.yaml

# 3) Finetune on GLUE (e.g., MNLI)
/usr/bin/python3 -m gpt1_factory.cli.finetune --cfg configs/finetune_glue.yaml data.task=mnli

# 4) Run ablations
/usr/bin/python3 -m gpt1_factory.cli.run_ablation --cfg configs/ablations.yaml
```

## Local corpus quickstart (train + generate)

Specify a small real checkpoint without downloading any external corpus. The
`sample_corpus.txt` file is copied into the local-text directory the trainer reads
from, then a small model is trained against a BPE trained on that corpus, and finally
the checkpoint generates continuation text.

```bash
/usr/bin/python3 -m pip install -r requirements.txt   # if needed
cp sample_corpus.txt gpt1_ablation_factory/data/text/
/usr/bin/python3 -m gpt1_factory.cli.pretrain --cfg configs/pretrain_local.yaml
/usr/bin/python3 -m gpt1_factory.cli.generate --bpe runs/bpe_local/bpe.json --ckpt runs/exp_local/checkpoints/latest.pt --model-yaml configs/model/gpt_mini.yaml --prompt "Once upon a time" --no-stop-on-eos --max-new-tokens 80
```

The small 4-layer model here overfits a ~100k-word corpus, so it is a demo of the
pipeline only, not a production GPT-1.

## Sample continuations

Verified continuation outputs from the trained local model (prompt
`Once upon a time there was a king who had three sons`):

```
Once upon a time there was a king who had three sons , and little is once a very the king , she was done with a beautiful little man on him
Once upon a time there was a king who had three sons , and it , and 'It heard the wolf was once to the sun .' Then the little man and of the water
```

## Notes

* Uses HuggingFace `datasets` for corpora: BookCorpusOpen / GLUE / RACE / StoryCloze.
* BPE training via `tokenizers` (40k merges by default).
* Checkpoints saved under `runs/exp_*/checkpoints/`.
* TensorBoard logs under `runs/exp_*/tb/`. Loguru writes `runs/exp_*/log.txt`.

### Local data cache

All datasets are cached under `gpt1_ablation_factory/data/hf_cache` by default.
You can prefetch everything:

```bash
bash scripts/download.sh
# or:
/usr/bin/python3 -m gpt1_factory.cli.download --target books
/usr/bin/python3 -m gpt1_factory.cli.download --target glue --glue-task mnli
/usr/bin/python3 -m gpt1_factory.cli.download --target race
/usr/bin/python3 -m gpt1_factory.cli.download --target story_cloze
```