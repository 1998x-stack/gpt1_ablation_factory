from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from ..configs import ExpConfig, FinetuneConfig
from ..models.checkpoint import load_pretrained_partial, save_checkpoint
from ..models.heads import (
    ChoiceScoringHead,
    ClassificationHead,
    pool_sequence_state,
)
from ..tasks.metrics import compute_metrics


class FinetuneTrainer:
    """Unified finetuning trainer with auxiliary LM and transfer-depth control."""

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

        if hasattr(backbone, "d_model"):
            d_model = int(backbone.d_model)
        elif hasattr(backbone, "tok_emb"):
            d_model = backbone.tok_emb.embedding_dim
        else:
            d_model = 768

        self.cls_head = ClassificationHead(
            d_model=d_model,
            num_labels=num_labels,
            dropout=cfg.head_dropout,
        ).to(self.device)
        self.choice_head = ChoiceScoringHead(
            d_model=d_model,
            dropout=cfg.head_dropout,
        ).to(self.device)

        if cfg.pretrained_path:
            info = load_pretrained_partial(
                self.backbone,
                cfg.pretrained_path,
                cfg.transfer_layers,
            )
            logger.info(
                "[finetune] loaded {} pretrained tensors; {} missing",
                len(info["loaded_keys"]),
                len(info["missing_keys"]),
            )

        self._trainable_params = (
            list(self.backbone.parameters())
            + list(self.cls_head.parameters())
            + list(self.choice_head.parameters())
        )
        self.optim = torch.optim.AdamW(
            self._trainable_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        total_steps = max(1, len(train_loader) * cfg.epochs)
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            # GPT-1 fine-tuning uses a warmup followed by linear decay.
            remaining = max(0, total_steps - step)
            decay_steps = max(1, total_steps - warmup_steps)
            return float(remaining) / float(decay_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda=lr_lambda,
        )
        self._is_regression = self.num_labels == 1

    def _forward_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        if input_ids.dim() == 3:
            B, C, L = input_ids.shape
            x = input_ids.view(B * C, L)
            attn = (
                attention_mask.view(B * C, L)
                if attention_mask is not None
                else None
            )
            out = self.backbone(input_ids=x, attention_mask=attn)
            return out, (B, C, L)

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out, None

    def _logits_for_mc(
        self,
        last_hidden_state: torch.Tensor,
        shape_tuple: tuple[int, int, int],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, C, L = shape_tuple
        flat_mask = (
            attention_mask.view(B * C, L)
            if attention_mask is not None
            else None
        )
        pooled = pool_sequence_state(
            last_hidden_state,
            attention_mask=flat_mask,
        )
        pooled = pooled.view(B, C, pooled.size(-1))
        return self.choice_head(pooled)

    def train(self) -> Tuple[float, dict]:
        global_step = 0
        best_metric = float("-inf")
        best_detail = {}

        for epoch in range(1, self.cfg.epochs + 1):
            self.backbone.train()
            self.cls_head.train()
            self.choice_head.train()

            for batch in self.train_loader:
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }

                with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                    out, shape_tuple = self._forward_backbone(
                        batch["input_ids"],
                        batch.get("attention_mask"),
                    )

                    if shape_tuple is not None:
                        logits = self._logits_for_mc(
                            out["last_hidden_state"],
                            shape_tuple,
                            batch.get("attention_mask"),
                        )
                        loss_cls = (
                            F.cross_entropy(logits, batch["labels"])
                            if "labels" in batch
                            else logits.mean() * 0.0
                        )
                    else:
                        logits = self.cls_head(
                            out["last_hidden_state"],
                            batch.get("attention_mask"),
                            batch.get("classify_positions"),
                        )
                        if self._is_regression:
                            loss_cls = F.mse_loss(
                                logits.squeeze(-1),
                                batch["labels"].float(),
                            )
                        else:
                            loss_cls = F.cross_entropy(
                                logits,
                                batch["labels"],
                            )

                    loss = loss_cls

                    if self.cfg.aux_lm_lambda > 0.0 and "labels_lm" in batch:
                        lm_ids = batch["input_ids"]
                        lm_lbl = batch["labels_lm"]
                        lm_attn = batch.get("attention_mask")

                        if lm_ids.dim() == 3:
                            B, C, L = lm_ids.shape
                            lm_ids = lm_ids.view(B * C, L)
                            lm_lbl = lm_lbl.view(B * C, L)
                            if lm_attn is not None:
                                lm_attn = lm_attn.view(B * C, L)

                        lm_out = self.backbone(
                            input_ids=lm_ids,
                            attention_mask=lm_attn,
                            labels=lm_lbl,
                        )
                        loss = (
                            loss
                            + self.cfg.aux_lm_lambda * lm_out["loss"]
                        )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params,
                    self.cfg.grad_clip,
                )
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                self.scheduler.step()

                global_step += 1
                if global_step % 50 == 0:
                    lr = self.optim.param_groups[0]["lr"]
                    logger.info(
                        "[finetune] epoch={} step={} lr={:.2e} loss={:.4f}",
                        epoch,
                        global_step,
                        lr,
                        loss.item(),
                    )

            metric_val, detail = self.evaluate()
            if metric_val > best_metric:
                best_metric = metric_val
                best_detail = detail
                save_checkpoint(
                    Path(self.exp.out_dir) / "checkpoints/best.pt",
                    self.backbone,
                    optim=self.optim,
                    step=global_step,
                    extra_modules={
                        "classification_head": self.cls_head,
                        "choice_head": self.choice_head,
                    },
                )

        return best_metric, best_detail

    @torch.no_grad()
    def evaluate(self) -> tuple[float, dict]:
        if self.valid_loader is None:
            raise ValueError("FinetuneTrainer requires a validation dataloader.")

        self.backbone.eval()
        self.cls_head.eval()
        self.choice_head.eval()
        ys, ps = [], []

        for batch in self.valid_loader:
            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }
            out, shape_tuple = self._forward_backbone(
                batch["input_ids"],
                batch.get("attention_mask"),
            )

            if shape_tuple is not None:
                logits = self._logits_for_mc(
                    out["last_hidden_state"],
                    shape_tuple,
                    batch.get("attention_mask"),
                )
                pred = torch.argmax(logits, dim=-1)
            else:
                logits = self.cls_head(
                    out["last_hidden_state"],
                    batch.get("attention_mask"),
                    batch.get("classify_positions"),
                )
                pred = (
                    logits.squeeze(-1)
                    if self._is_regression
                    else torch.argmax(logits, dim=-1)
                )

            if "labels" in batch:
                ys.extend(batch["labels"].cpu().tolist())
                ps.extend(pred.cpu().tolist())

        metrics = (
            compute_metrics(self.task_name, ys, ps)
            if ys
            else {"acc": 0.0}
        )
        score = list(metrics.values())[0] if metrics else 0.0
        logger.info("[eval-{}] {}", self.task_name, metrics)
        return float(score), metrics
