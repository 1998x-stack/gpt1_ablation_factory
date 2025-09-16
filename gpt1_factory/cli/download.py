from __future__ import annotations

import argparse
from pathlib import Path

from ..data.downloader import (
    download_books_like, download_glue, download_race, download_story_cloze
)

DEF_CACHE = Path("gpt1_ablation_factory/data/hf_cache")
DEF_EXPORT = Path("gpt1_ablation_factory/data/exports")
DEF_TEXT   = Path("gpt1_ablation_factory/data/text")


def main():
    parser = argparse.ArgumentParser(description="Download datasets into local project data/ folder.")
    parser.add_argument("--target", type=str, default="all",
                        choices=["all", "books", "glue", "race", "story_cloze"])
    parser.add_argument("--glue-task", type=str, default="mnli",
                        help="When target=glue, which task to fetch")
    parser.add_argument("--cache-dir", type=str, default=str(DEF_CACHE))
    parser.add_argument("--export-dir", type=str, default=str(DEF_EXPORT))
    parser.add_argument("--local-text-dir", type=str, default=str(DEF_TEXT),
                        help="If provided and contains .txt files, used for books-like pretraining corpus")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    export_dir = Path(args.export_dir)
    local_text_dir = Path(args.local_text_dir) if args.local_text_dir else None

    if args.target in ("all", "books"):
        download_books_like(cache_dir, export_dir, local_text_dir)

    if args.target in ("all", "glue"):
        download_glue(args.glue_task, cache_dir, export_dir)

    if args.target in ("all", "race"):
        download_race(cache_dir, export_dir)

    if args.target in ("all", "story_cloze"):
        download_story_cloze(cache_dir, export_dir)


if __name__ == "__main__":
    main()
