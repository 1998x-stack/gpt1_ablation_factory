"""Create or execute paper-driven, isolated ablation jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ..experiments.factory import build_plan, run_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-1 ablation factory")
    parser.add_argument("--cfg", default="configs/paper_ablations.yaml")
    parser.add_argument("--execute", action="store_true", help="Run jobs; default is dry-run")
    parser.add_argument("--resume", action="store_true", help="Skip completed jobs with matching config")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", help="Optional JSON plan/result output")
    args = parser.parse_args()
    jobs = build_plan(args.cfg)
    results = run_plan(jobs, execute=args.execute, resume=args.resume,
                       fail_fast=args.fail_fast)
    payload = {"count": len(jobs), "results": results}
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.execute and any(r["status"] not in {"completed", "skipped"} for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
