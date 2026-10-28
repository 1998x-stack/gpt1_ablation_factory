"""Create offline interactive HTML / CSV / JSON ablation reports."""
from __future__ import annotations
import argparse
import json

from ..experiments.report import write_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/paper_ablations")
    parser.add_argument("--out", default="runs/ablation_report")
    args = parser.parse_args()
    print(json.dumps(write_report(args.runs, args.out), indent=2))


if __name__ == "__main__":
    main()
