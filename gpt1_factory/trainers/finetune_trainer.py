from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from ..configs import ExpConfig, FinetuneConfig
from ..models.gpt_decoder import GPTClassificationHead
from ..models.heads import ClassificationHead
from ..models.checkpoint import load_pretrained_partial, save_checkpoint
from ..tasks.metrics import compute_metrics


class FinetuneTrainer:
    """统一微调 Trainer：支持辅助LM损失、层数迁移、分类头。"""

    def __init__(
        self,
        exp: ExpConfig,
        cfg: FinetuneConfig,
        backbone: torch.nn.Module,
        num_labels: int,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader],
        task_name: str,
    ) -> None:
        self.exp = exp
        self.cfg = cfg
        self.backbone = backbone
        self.num_labels = num_labels
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.task_name = task_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)

        self.cls_head = ClassificationHead(d_model=backbone.ln_f.normalized_shape[0] if hasattr(backbone, "ln_f") else 768,
                                           num_labels=num_labels, dropout=cfg.head_dropout).to(self.device)

        if cfg.pretrained_path:
            load_pretrained_partial(self.backbone, cfg.pretrained_path, cfg.transfer_layers)

        # 仅分类头 + 部分下层可选择性解冻，这里默认全部训练（论文微调设置简洁）
        self.optim = torch.optim.AdamW(list(self.backbone.parameters()) + list(self.cls_head.parameters()),
                                       lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    def _compute_aux_lm_loss(self, logits, labels) -> torch.Tensor:
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)

    def train(self) -> Tuple[float, dict]:
        global_step = 0
        best_metric = -1.0
        best_state = None

        for epoch in range(1, self.cfg.epochs + 1):
            self.backbone.train()
            self.cls_head.train()
            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
                    logits_cls = self.cls_head(out["last_hidden_state"], batch.get("attention_mask"))
                    loss_cls = F.cross_entropy(logits_cls, batch["labels"])
                    loss = loss_cls
                    # 可选：辅助 LM loss（需要 labels，即下一个 token 任务）
                    if self.cfg.aux_lm_lambda > 0.0 and "labels_lm" in batch:
                        lm_out = self.backbone(input_ids=batch["input_ids"], labels=batch["labels_lm"])
                        loss = loss + self.cfg.aux_lm_lambda * lm_out["loss"]

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

                global_step += 1
                if global_step % 50 == 0:
                    logger.info(f"[finetune] epoch={epoch} step={global_step} loss={loss.item():.4f}")

            # 验证
            metric_val, detail = self.evaluate()
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = {
                    "backbone": self.backbone.state_dict(),
                    "cls_head": self.cls_head.state_dict(),
                }
                save_checkpoint(Path(self.exp.out_dir) / "checkpoints/best.pt", self.backbone)

        return best_metric, detail

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict]:
        self.backbone.eval()
        self.cls_head.eval()
        ys, ps = [], []
        for batch in self.valid_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            logits = self.cls_head(out["last_hidden_state"], batch.get("attention_mask"))
            pred = torch.argmax(logits, dim=-1)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(pred.cpu().tolist())
        metrics = compute_metrics(self.task_name, ys, ps)
        score = list(metrics.values())[0]
        logger.info(f"[eval-{self.task_name}] {metrics}")
        return float(score), metrics
