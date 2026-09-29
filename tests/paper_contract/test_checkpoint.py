import torch
import torch.nn as nn

from gpt1_factory.models.checkpoint import save_checkpoint


def test_checkpoint_persists_task_modules(tmp_path):
    backbone = nn.Linear(4, 4)
    head = nn.Linear(4, 2)
    path = tmp_path / "best.pt"

    save_checkpoint(
        path,
        backbone,
        step=7,
        extra_modules={"classification_head": head},
    )

    payload = torch.load(path, map_location="cpu")

    assert payload["step"] == 7
    assert "classification_head" in payload["modules"]
    assert set(payload["modules"]["classification_head"]) == set(head.state_dict())
