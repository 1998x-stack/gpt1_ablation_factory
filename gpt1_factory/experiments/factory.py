"""Validated GPT-1 experiment factory: configurations are products, not mutable globals."""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from ..configs import DataConfig, ExpConfig, FinetuneConfig, ModelConfig
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[2]
SAFE_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def deep_merge(base: dict, patch: Mapping[str, Any]) -> dict:
    """Merge nested configuration sections without losing siblings."""
    merged = {key: (deep_merge(value, {}) if isinstance(value, dict) else value)
              for key, value in base.items()}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, seen: frozenset[Path] = frozenset()) -> dict:
    """Load YAML includes in order; reject cycles and missing include paths."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Include cycle: {path}")
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    includes = cfg.pop("include", []) or []
    if not isinstance(includes, list):
        raise ValueError(f"include must be a list: {path}")
    resolved: dict = {}
    for item in includes:
        inc = Path(item)
        if not inc.is_absolute():
            inc = ROOT / inc
        resolved = deep_merge(resolved, load_config(inc, seen | {path}))
    return deep_merge(resolved, cfg)


def _identifier(value: str) -> str:
    value = str(value)
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"Invalid experiment identifier: {value!r}")
    return value


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _digest(cfg: dict) -> str:
    value = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Job:
    name: str
    suite: str
    variant: str
    task: str
    seed: int
    out_dir: Path
    config: dict
    config_hash: str


def build_plan(spec_path: str | Path) -> list[Job]:
    """Expand suite x variant x task x seed x transfer-depth into isolated jobs."""
    spec = load_config(spec_path)
    if spec.get("schema_version") != 1:
        raise ValueError("Ablation manifest requires schema_version: 1")
    base_path = spec["base_config"]
    base = load_config(_path(base_path))
    default_model = spec.get("default_model_config")
    artifact = spec.get("tokenizer_artifact")
    if not artifact:
        raise ValueError("tokenizer_artifact is mandatory for all arms")
    runs_dir = _path(spec.get("runs_dir", "runs/paper_ablations")).resolve()
    jobs: list[Job] = []
    names: set[str] = set()
    for suite in spec.get("suites", []):
        suite_id = _identifier(suite["id"])
        tasks = suite.get("tasks", ["sst2"])
        seeds = suite.get("seeds", [42])
        variants = suite.get("variants", [])
        for arm in variants:
            arm_id = _identifier(arm["id"])
            depths = arm.get("transfer_layers", [-1])
            lambdas = arm.get("aux_lm_lambda", [0.5])
            for task, seed, depth, lam in itertools.product(tasks, seeds, depths, lambdas):
                task = _identifier(task)
                if task not in {"cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"}:
                    raise ValueError(f"Unsupported GLUE task in factory: {task}")
                if int(depth) < -1 or float(lam) < 0:
                    raise ValueError("transfer_layers >= -1 and aux_lm_lambda >= 0 required")
                model_path = arm.get("model_config", default_model)
                if not model_path:
                    raise ValueError(f"Missing model_config: {arm_id}")
                model = load_config(_path(model_path))
                cfg = deep_merge(base, model)
                cfg = deep_merge(cfg, arm.get("overrides", {}))
                backbone = cfg.get("model", {}).get("name")
                no_pretrain = bool(arm.get("no_pretrain", False))
                if backbone == "lstm_baseline" and int(depth) != -1:
                    raise ValueError("LSTM does not support Transformer transfer-depth ablation")
                if backbone not in {"gpt_decoder", "lstm_baseline"}:
                    raise ValueError(f"Unknown backbone: {backbone}")
                pretrained = "" if no_pretrain else arm.get("pretrained_path", spec.get("pretrained_path", ""))
                if not no_pretrain and not pretrained:
                    raise ValueError(f"{arm_id} requires its own pretrained checkpoint")
                if backbone == "lstm_baseline" and not no_pretrain and pretrained == spec.get("pretrained_path"):
                    raise ValueError("LSTM must use its separately pretrained LSTM checkpoint")
                tag = f"{suite_id}__{arm_id}__{task}__k{depth}__lam{str(lam).replace('.', 'p')}__s{seed}"
                if tag in names:
                    raise ValueError(f"Duplicate run: {tag}")
                names.add(tag)
                cfg = deep_merge(cfg, {
                    "exp": {"seed": int(seed), "out_dir": str(runs_dir / tag)},
                    "data": {"name": "glue", "task": task, "tokenizer_artifact": str(_path(artifact))},
                    "finetune": {
                        "pretrained_path": str(_path(pretrained)) if pretrained else "",
                        "aux_lm_lambda": float(lam),
                        "transfer_layers": int(depth),
                    },
                })
                for section, schema in (
                    ("exp", ExpConfig), ("data", DataConfig),
                    ("finetune", FinetuneConfig), ("model", ModelConfig),
                ):
                    valid = {f.name for f in fields(schema)}
                    unknown = set(cfg.get(section, {})) - valid
                    if unknown:
                        raise ValueError(
                            f"Unknown {section} configuration keys for {tag}: {sorted(unknown)}"
                        )
                digest = _digest(cfg)
                jobs.append(Job(tag, suite_id, arm_id, task, int(seed),
                                runs_dir / tag, cfg, digest))
    if not jobs:
        raise ValueError("Ablation manifest expanded to zero jobs")
    return jobs


def _run_one(job: Job, *, resume: bool) -> dict:
    out = job.out_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.json"
    provenance = out / "provenance.json"
    if provenance.exists():
        prev = json.loads(provenance.read_text(encoding="utf-8"))
        if prev.get("config_hash") != job.config_hash:
            raise ValueError(f"Refusing to overwrite a different experiment: {out}")
        if resume and metrics.exists() and prev.get("status") == "completed":
            return {"run": job.name, "status": "skipped"}
    ckpt = job.config["finetune"]["pretrained_path"]
    tok = job.config["data"]["tokenizer_artifact"]
    if ckpt and not Path(ckpt).is_file():
        raise FileNotFoundError(f"Missing checkpoint for {job.name}: {ckpt}")
    if not Path(tok).exists():
        raise FileNotFoundError(f"Missing tokenizer artifact for {job.name}: {tok}")
    resolved_cfg = out / "resolved.yaml"
    resolved_cfg.write_text(yaml.safe_dump(job.config, allow_unicode=True,
                                          sort_keys=True), encoding="utf-8")
    record = {
        "run": job.name, "suite": job.suite, "variant": job.variant,
        "task": job.task, "seed": job.seed, "config_hash": job.config_hash,
        "status": "running", "checkpoint": ckpt, "tokenizer_artifact": tok,
        "note": "Paths are recorded; exact corpus identity requires external dataset manifests.",
    }
    provenance.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, "-m", "gpt1_factory.cli.finetune", "--cfg", str(resolved_cfg)]
    with (out / "runner.log").open("w", encoding="utf-8") as stream:
        try:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            record["status"] = "completed" if metrics.exists() else "missing_metrics"
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["returncode"] = exc.returncode
        finally:
            provenance.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    return {"run": job.name, "status": record["status"]}


def run_plan(jobs: list[Job], *, execute: bool, resume: bool = False,
             fail_fast: bool = False) -> list[dict]:
    """Dry-run is side-effect-free. Execute requires all upstream artifacts."""
    if not execute:
        return [{"run": j.name, "status": "planned", "config_hash": j.config_hash,
                 "out_dir": str(j.out_dir)} for j in jobs]
    results = []
    for job in jobs:
        try:
            item = _run_one(job, resume=resume)
        except (ValueError, FileNotFoundError) as exc:
            item = {"run": job.name, "status": "blocked", "error": str(exc)}
        results.append(item)
        if fail_fast and item["status"] not in {"completed", "skipped"}:
            break
    return results
