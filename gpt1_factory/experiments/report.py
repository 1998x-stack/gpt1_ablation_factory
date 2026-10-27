"""Aggregate completed experiment artifacts without inventing missing results."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path


def collect_runs(root: str | Path) -> list[dict]:
    rows = []
    for meta_path in sorted(Path(root).glob("*/provenance.json")):
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        metrics_path = meta_path.parent / "metrics.json"
        cfg_path = meta_path.parent / "resolved.yaml"
        if data.get("status") != "completed" or not metrics_path.is_file() or not cfg_path.is_file():
            continue
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        metric = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = float(metric["score"])
        if not math.isfinite(score):
            continue
        rows.append({
            "run": data["run"],
            "suite": data["suite"], "variant": data["variant"], "task": data["task"],
            "seed": int(data["seed"]), "depth": int(cfg["finetune"]["transfer_layers"]),
            "lambda": float(cfg["finetune"]["aux_lm_lambda"]),
            "metric": metric["primary_metric"], "score": score,
            "step": int(metric["step"]), "best_epoch": int(metric["best_epoch"]),
        })
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["suite"], row["variant"], row["task"],
               row["depth"], row["lambda"], row["metric"])
        groups[key].append(row["score"])
    out = []
    for (suite, variant, task, depth, lam, metric), values in sorted(groups.items()):
        out.append({
            "suite": suite, "variant": variant, "task": task,
            "depth": depth, "lambda": lam, "metric": metric,
            "seeds": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
        })
    return out


def paired_deltas(rows: list[dict], base: str = "full") -> list[dict]:
    """Difference to control only for matched (suite, task, seed, depth, lambda)."""
    lookups = {(r["suite"], r["variant"], r["task"], r["seed"], r["depth"], r["lambda"]): r
               for r in rows}
    deltas = defaultdict(list)
    for r in rows:
        if r["variant"] == base:
            continue
        ref = lookups.get((r["suite"], base, r["task"], r["seed"], r["depth"], r["lambda"]))
        if ref and ref["metric"] == r["metric"]:
            deltas[(r["suite"], r["variant"], r["task"], r["metric"])].append(r["score"] - ref["score"])
    return [{"suite": k[0], "variant": k[1], "task": k[2], "metric": k[3],
             "n_pairs": len(v), "delta_mean": statistics.mean(v),
             "delta_std": statistics.stdev(v) if len(v) > 1 else None}
            for k, v in sorted(deltas.items())]


def write_report(runs_dir: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_runs(runs_dir)
    stats = aggregate(rows)
    paired = paired_deltas(rows)
    (out / "summary.json").write_text(json.dumps(
        {"runs": rows, "groups": stats, "paired_deltas": paired},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for filename, dataset in (("runs.csv", rows), ("groups.csv", stats), ("paired_deltas.csv", paired)):
        if dataset:
            with (out / filename).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=dataset[0].keys())
                w.writeheader()
                w.writerows(dataset)
    from .plots import render_figures
    figures = render_figures(stats, out / "figures")
    dashboard = _dashboard(stats, paired, figures)
    (out / "dashboard.html").write_text(dashboard, encoding="utf-8")
    return {"completed_runs": len(rows), "groups": len(stats),
            "figures": figures, "dashboard": str(out / "dashboard.html")}


def _dashboard(stats: list[dict], paired: list[dict], figures: list[str]) -> str:
    """Self-contained HTML: interactive filters and SVG-free table fallback."""
    import json
    stats_json = json.dumps(stats, ensure_ascii=False, allow_nan=False)
    paired_json = json.dumps(paired, ensure_ascii=False, allow_nan=False)
    gallery = '<h2>Task-specific figures</h2>' + ''.join(
        '<figure><img alt="Task-specific research plot" loading="lazy" '
        'style="max-width:100%;width:900px" src="figures/' + escape(Path(p).name) + '">'
        '<figcaption>' + escape(Path(p).name) + '</figcaption></figure>'
        for p in figures
    )
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>GPT-1 Ablation Research Dashboard</title>
<style>
body{font:15px system-ui,sans-serif;max-width:1300px;margin:28px auto;padding:0 20px;color:#1f2937}
h1,h2{letter-spacing:-.03em}select{padding:8px;border:1px solid #adb5bd;border-radius:8px;margin-right:14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0 30px}
td,th{text-align:left;padding:9px;border-bottom:1px solid #e4e7eb}
th{background:#f7f8fa;position:sticky;top:0}
small{color:#657083} #plot{height:420px;overflow:auto}
.bar{height:20px;background:#64748b;border-radius:3px;min-width:1px}
table tr:hover{background:#f5f5f5}
</style>
<h1>GPT-1 paper-driven ablations</h1>
<p>Only completed runs are included. Mean ± sample SD are unitless task scores
(accuracy, F1, Pearson, or Matthews); do not average across metric types.</p>
<label>Suite <select id="suite"></select></label>
<label>Task <select id="task"></select></label>
<h2>Group comparison</h2><div id="plot"></div>
<h2>Paired differences to full</h2>
<p><small>Only trials sharing task, seed, depth and LM weight are paired.</small></p>
<div id="deltas"></div>
<script>
const stats = """ + stats_json + """;
const pairs = """ + paired_json + """;
function options(id, vals) {
 const sel=document.getElementById(id);
 sel.innerHTML='<option value="*">All</option>'+[...new Set(vals)].sort().map(x=>'<option>'+x+'</option>').join('');
 sel.addEventListener('change',render);
}
function table(rows,cols){
 if(!rows.length)return '<p>No completed matching runs.</p>';
 return '<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+
 rows.map(r=>'<tr>'+cols.map(c=>'<td>'+String(r[c]===null?'—':r[c]).replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
function render(){
 const suite=document.getElementById('suite').value,task=document.getElementById('task').value;
 const match=r=>(suite==='*'||suite===r.suite)&&(task==='*'||task===r.task);
 const data=stats.filter(match).map(r=>({...r,mean:r.mean.toFixed(4),
 std:r.std===null?'—':r.std.toFixed(4)}));
 document.getElementById('plot').innerHTML=table(data,['suite','variant','task','depth','lambda','metric','seeds','mean','std']);
 const d=pairs.filter(match).map(r=>({...r,delta_mean:r.delta_mean.toFixed(4),
 delta_std:r.delta_std===null?'—':r.delta_std.toFixed(4)}));
 document.getElementById('deltas').innerHTML=table(d,['suite','variant','task','metric','n_pairs','delta_mean','delta_std']);
}
options('suite',stats.map(x=>x.suite));options('task',stats.map(x=>x.task));render();
</script></html>"""
    return html.replace("</script></html>", "</script>" + gallery + "</html>")
