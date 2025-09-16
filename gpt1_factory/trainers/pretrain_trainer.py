from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from loguru import logger

from ..configs import ExpConfig, OptimConfig, DataConfig
from ..utils.tensorboard import create_tb_writer
from ..utils.schedules import linear_warmup_cosine_decay
from ..models.checkpoint import save_checkpoint


class PretrainTrainer:
    """语言模型预训练 Trainer。"""

    def __init__(self, exp: ExpConfig, optim_cfg: OptimConfig, model: torch.nn.Module,
                 train_loader: DataLoader, out_dir: str | Path, amp: bool = True) -> None:
        self.exp = exp
        self.optim_cfg = optim_cfg
        self.model = model
        self.train_loader = train_loader
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.writer = create_tb_writer(self.out_dir)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=optim_cfg.lr,
                                       betas=optim_cfg.betas, weight_decay=optim_cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, linear_warmup_cosine_decay(optim_cfg.warmup_steps, optim_cfg.max_steps)
        )

    def train(self, save_every: int = 10000, keep_last: int = 5) -> None:
        step = 0
        losses = []
        for batch in self.train_loader:
            self.model.train()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model(**batch)
                loss = out["loss"]
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optim_cfg.grad_clip)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            self.scheduler.step()

            step += 1
            losses.append(loss.item())
            if step % 100 == 0:
                avg = sum(losses[-100:]) / min(100, len(losses))
                self.writer.add_scalar("train/loss", avg, step)
                self.writer.add_scalar("train/lr", self.optim.param_groups[0]["lr"], step)
                logger.info(f"[pretrain] step={step} loss={avg:.4f}")

            if step % save_every == 0:
                save_checkpoint(self.out_dir / f"checkpoints/step_{step}.pt", self.model, self.optim, step)
                save_checkpoint(self.out_dir / f"checkpoints/latest.pt", self.model, self.optim, step)

            if step >= self.optim_cfg.max_steps:
                logger.info("Pretraining finished.")
                break
