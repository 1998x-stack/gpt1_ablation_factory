from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger

from ..configs import ExpConfig, FinetuneConfig
from ..models.heads import ClassificationHead
from ..models.checkpoint import load_pretrained_partial, save_checkpoint
from ..tasks.metrics import compute_metrics


class FinetuneTrainer:
    """统一微调 Trainer：支持分类/回归/多选，辅助LM，线性warmup→常数。"""

    def __init__(self, exp: ExpConfig, cfg: FinetuneConfig, backbone: torch.nn.Module, num_labels: int,
                 train_loader: DataLoader, valid_loader: Optional[DataLoader], task_name: str) -> None:
        self.exp = exp
        self.cfg = cfg
        self.backbone = backbone
        self.num_labels = num_labels
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.task_name = task_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone.to(self.device)

        if hasattr(backbone, "ln_f"):
            d_model = backbone.ln_f.normalized_shape[0]
        elif hasattr(backbone, "tok_emb"):
            d_model = backbone.tok_emb.embedding_dim
        else:
            d_model = 768

        self.cls_head = ClassificationHead(d_model=d_model, num_labels=num_labels, dropout=cfg.head_dropout).to(self.device)

        if cfg.pretrained_path:
            load_pretrained_partial(self.backbone, cfg.pretrained_path, cfg.transfer_layers)

        params = list(self.backbone.parameters()) + list(self.cls_head.parameters())
        self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        total_steps = max(1, len(train_loader) * cfg.epochs)
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)
        self._is_regression = (self.num_labels == 1)

    def _forward_backbone(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            x = input_ids.view(B * C, L)
            attn = attention_mask.view(B * C, L) if attention_mask is not None else None
            out = self.backbone(input_ids=x, attention_mask=attn)
            return out, (B, C, L)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            return out, None

    def _logits_for_mc(self, last_hidden_state, shape_tuple):
        B, C, L = shape_tuple
        H = last_hidden_state.size(-1)
        x = last_hidden_state.view(B, C, L, H)[:, :, -1, :]
        x = self.cls_head.drop(x)
        logits = torch.stack([self.cls_head.fc(x[:, i, :]) for i in range(C)], dim=1).squeeze(-1)
        return logits

    def train(self) -> Tuple[float, dict]:
        global_step = 0
        best_metric = -1.0
        best_detail = {}

        for epoch in range(1, self.cfg.epochs + 1):
            self.backbone.train()
            self.cls_head.train()

            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    out, shape_tuple = self._forward_backbone(batch["input_ids"], batch.get("attention_mask"))

                    if shape_tuple is not None:
                        logits = self._logits_for_mc(out["last_hidden_state"], shape_tuple)
                        loss_cls = F.cross_entropy(logits, batch["labels"]) if "labels" in batch else logits.mean() * 0.0
                    else:
                        feats = out["last_hidden_state"]
                        logits = self.cls_head(feats, batch.get("attention_mask"))
                        if self._is_regression:
                            loss_cls = F.mse_loss(logits.squeeze(-1), batch["labels"].float())
                        else:
                            loss_cls = F.cross_entropy(logits, batch["labels"])

                    loss = loss_cls

                    # —— 修复：多选时对 LM 辅助也需展平 (B,C,L) → (B*C,L)
                    if self.cfg.aux_lm_lambda > 0.0 and "labels_lm" in batch:
                        if batch["input_ids"].dim() == 3:
                            B, C, L = batch["input_ids"].shape
                            lm_ids = batch["input_ids"].view(B * C, L)
                            lm_lbl = batch["labels_lm"].view(B * C, L)
                        else:
                            lm_ids = batch["input_ids"]
                            lm_lbl = batch["labels_lm"]
                        lm_out = self.backbone(input_ids=lm_ids, labels=lm_lbl)
                        loss = loss + self.cfg.aux_lm_lambda * lm_out["loss"]

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                self.scheduler.step()

                global_step += 1
                if global_step % 50 == 0:
                    lr = self.optim.param_groups[0]["lr"]
                    logger.info(f"[finetune] epoch={epoch} step={global_step} lr={lr:.2e} loss={loss.item():.4f}")

            metric_val, detail = self.evaluate()
            if metric_val > best_metric:
                best_metric, best_detail = metric_val, detail
                save_checkpoint(Path(self.exp.out_dir) / "checkpoints/best.pt", self.backbone)

        return best_metric, best_detail

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict]:
        self.backbone.eval()
        self.cls_head.eval()
        ys, ps = [], []

        for batch in self.valid_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out, shape_tuple = self._forward_backbone(batch["input_ids"], batch.get("attention_mask"))

            if shape_tuple is not None:
                logits = self._logits_for_mc(out["last_hidden_state"], shape_tuple)
                pred = torch.argmax(logits, dim=-1)
            else:
                feats = out["last_hidden_state"]
                logits = self.cls_head(feats, batch.get("attention_mask"))
                pred = logits.squeeze(-1) if self._is_regression else torch.argmax(logits, dim=-1)

            if "labels" in batch:
                ys.extend(batch["labels"].cpu().tolist())
                ps.extend(pred.cpu().tolist())

        metrics = compute_metrics(self.task_name, ys, ps) if ys else {"acc": 0.0}
        score = list(metrics.values())[0] if metrics else 0.0
        logger.info(f"[eval-{self.task_name}] {metrics}")
        return float(score), metrics
