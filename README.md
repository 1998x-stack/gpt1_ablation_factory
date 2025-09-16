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
│  ├─ finetune_glue.yaml
│  ├─ ablations.yaml
│  ├─ model/
│  │  ├─ gpt_small.yaml
│  │  └─ lstm_baseline.yaml
│  └─ data/
│     ├─ books_corpus_open.yaml
│     ├─ glue_mnli.yaml
│     ├─ glue_sst2.yaml
│     ├─ race.yaml
│     └─ story_cloze.yaml
├─ scripts/
│  ├─ pretrain.sh
│  ├─ finetune.sh
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
│     ├─ finetune.py
│     └─ run_ablation.py
└─ tests/
   ├─ test_tokenizer.py
   ├─ test_models.py
   └─ test_collators.py
```

## Quickstart

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .

# 2) Pretrain (BooksCorpusOpen; BPE trained on-the-fly)
python -m gpt1_factory.cli.pretrain --cfg configs/pretrain_books.yaml

# 3) Finetune on GLUE (e.g., MNLI)
python -m gpt1_factory.cli.finetune --cfg configs/finetune_glue.yaml data.task=mnli

# 4) Run ablations
python -m gpt1_factory.cli.run_ablation --cfg configs/ablations.yaml
````

## Notes

* Uses HuggingFace `datasets` for corpora: BookCorpusOpen / GLUE / RACE / StoryCloze.
* BPE training via `tokenizers` (40k merges by default).
* Checkpoints saved under `runs/exp_*/checkpoints/`.
* TensorBoard logs under `runs/exp_*/tb/`. Loguru writes `runs/exp_*/log.txt`.