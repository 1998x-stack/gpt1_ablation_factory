"""Prevent documentation and shell-entrypoint drift."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_root_docs_have_resolving_local_links():
    for source in (ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "docs/README.md"):
        assert source.is_file()
        text = source.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            path = (source.parent / target.split("#", 1)[0]).resolve()
            assert path.exists(), f"Broken link from {source}: {target}"


def test_shell_wrappers_use_virtualenv_python_and_parse():
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "/usr/bin/python3" not in text, script
        assert "set -euo pipefail" in text, script
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_agent_contract_and_demo_are_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for path in ("configs/ablations.yaml", "configs/paper_ablations.yaml",
                 "configs/finetune_smoke.yaml", "docs/PAPER_ABLATION_FACTORY.md"):
        assert (ROOT / path).exists(), path
        assert path in readme
    assert "docs/TOKENIZER_TASK_PROTOCOL.md" in agents
    assert "reproduced paper results" in agents or "reproduced paper results" in readme
