from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Iterable, List, Dict

import datasets


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _export_jsonl(ds: datasets.Dataset, out_path: Path, text_fields: Tuple[str, str | None] | None) -> None:
    """Export to JSONL so you can go fully offline later."""
    _ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in ds:
            if text_fields is None:
                txt = rec.get("text", "")
            else:
                a, b = text_fields
                t1 = rec[a] if a else ""
                t2 = rec[b] if b else ""
                txt = f"{t1}\n{t2}"
            f.write({"text": txt}.__repr__() + "\n")  # lightweight JSONL; replace with json.dumps if you prefer


def download_books_like(cache_dir: Path, export_dir: Optional[Path] = None, local_text_dir: Optional[Path] = None) -> None:
    cache_dir = _ensure_dir(cache_dir)
    if local_text_dir and local_text_dir.exists():
        ds = datasets.load_dataset("text", data_files={"train": str(local_text_dir / "**/*.txt")}, split="train", cache_dir=str(cache_dir))
    else:
        try:
            ds = datasets.load_dataset("hf://datasets/bookcorpusopen/bookcorpusopen", split="train", cache_dir=str(cache_dir))
        except Exception:
            try:
                ds = datasets.load_dataset("Skylion007/openwebtext", split="train", cache_dir=str(cache_dir))
            except Exception:
                ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train", cache_dir=str(cache_dir))
    if export_dir:
        _export_jsonl(ds, _ensure_dir(export_dir) / "books_like_train.jsonl", None)


def download_glue(task: str, cache_dir: Path, export_dir: Optional[Path] = None) -> None:
    cache_dir = _ensure_dir(cache_dir)
    raw = datasets.load_dataset("glue", task, cache_dir=str(cache_dir))
    if export_dir:
        ed = _ensure_dir(export_dir)
        if task == "sst2":
            tf = ("sentence", None)
        elif task == "mnli":
            tf = ("premise", "hypothesis")
        elif task in ("mrpc", "qqp"):
            tf = ("sentence1", "sentence2")
        elif task == "cola":
            tf = ("sentence", None)
        elif task == "stsb":
            tf = ("sentence1", "sentence2")
        else:
            tf = ("sentence", None)
        for split in ["train", "validation", "test", "validation_matched", "test_matched"]:
            if split in raw:
                _export_jsonl(raw[split], ed / f"glue_{task}_{split}.jsonl", tf)


def download_race(cache_dir: Path, export_dir: Optional[Path] = None) -> None:
    cache_dir = _ensure_dir(cache_dir)
    raw = datasets.load_dataset("race", "all", cache_dir=str(cache_dir))
    if export_dir:
        ed = _ensure_dir(export_dir)
        for split in ["train", "validation", "test"]:
            if split in raw:
                _export_jsonl(raw[split], ed / f"race_{split}.jsonl", ("article", "question"))


def download_story_cloze(cache_dir: Path, export_dir: Optional[Path] = None) -> None:
    cache_dir = _ensure_dir(cache_dir)
    raw = datasets.load_dataset("story_cloze", "2016", cache_dir=str(cache_dir))
    if export_dir:
        ed = _ensure_dir(export_dir)
        for split in ["validation", "test"]:
            _export_jsonl(raw[split], ed / f"storycloze_{split}.jsonl", None)
