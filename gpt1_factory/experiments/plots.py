"""Paper-oriented charts; each task retains its own metric scale."""
from __future__ import annotations

from pathlib import Path


def render_figures(groups: list[dict], out_dir: str | Path) -> list[str]:
    if not groups:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = Path(out_dir)
    figures.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    tasks = sorted({g["task"] for g in groups})
    for task in tasks:
        selected = [g for g in groups if g["task"] == task and g["suite"] == "paper_table5"]
        if selected:
            selected.sort(key=lambda g: g["variant"])
            fig, ax = plt.subplots(figsize=(9, 4.8))
            xs = list(range(len(selected)))
            ax.bar(xs, [g["mean"] for g in selected], yerr=[
                g["std"] if g["std"] is not None else 0 for g in selected
            ], capsize=4)
            ax.set_xticks(xs, [g["variant"] for g in selected], rotation=25)
            ax.set_ylabel(selected[0]["metric"])
            ax.set_title(f"GPT-1 Table 5 ablations — {task.upper()} (repo measurements)")
            ax.grid(axis="y", alpha=0.18)
            fig.tight_layout()
            p = figures / f"table5_{task}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            saved.append(str(p))
        transfer = [g for g in groups if g["task"] == task and
                    g["suite"] == "paper_transfer" and g["depth"] >= 0]
        if transfer:
            transfer.sort(key=lambda g: g["depth"])
            fig, ax = plt.subplots(figsize=(8.5, 4.6))
            xs = [g["depth"] for g in transfer]
            ys = [g["mean"] for g in transfer]
            err = [g["std"] if g["std"] is not None else 0 for g in transfer]
            ax.errorbar(xs, ys, yerr=err, marker="o", capsize=4)
            ax.set_xlabel("Transferred Transformer blocks (K=0: embeddings only)")
            ax.set_ylabel(transfer[0]["metric"])
            ax.set_title(f"Transfer-depth study — {task.upper()} (repo measurements)")
            ax.set_xticks(xs)
            ax.grid(alpha=0.18)
            fig.tight_layout()
            p = figures / f"transfer_{task}.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            saved.append(str(p))
    return saved
