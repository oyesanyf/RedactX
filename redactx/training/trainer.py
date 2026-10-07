"""
Optimization Engine for RedactX: Multi-Task Decision & Span Attribution Fine-Tuning.
Includes differential learning rates, gradient clipping, evaluation metrics (F1, Recall, AUC),
and post-training Platt temperature calibration.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from redactx.config import TrainingConfig
from redactx.data.dataset import RedactXDataset
from redactx.models.engine import RedactXDecisionEngine
from redactx.training.losses import RLCDCalibrationLoss

logger = logging.getLogger("redactx.trainer")


class RedactXTrainer:
    """
    High-level trainer managing fine-tuning across LoRA adapters, Hobson pointer head,
    and span locator.
    """

    def __init__(
        self,
        engine: RedactXDecisionEngine,
        config: Optional[TrainingConfig] = None
    ):
        self.engine = engine
        self.config = config or TrainingConfig()
        self.device = engine.device

    def train(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training and validation loop.
        """
        num_epochs = epochs or self.config.epochs
        bs = batch_size or self.config.batch_size
        save_dir = output_dir or self.config.output_dir

        train_dataset = RedactXDataset(
            train_records,
            self.engine.tokenizer,
            max_length=self.engine.config.max_length,
            include_span_labels=self.engine.config.enable_span_locator
        )
        val_dataset = RedactXDataset(
            val_records,
            self.engine.tokenizer,
            max_length=self.engine.config.max_length,
            include_span_labels=self.engine.config.enable_span_locator
        )

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

        # Differential Learning Rates
        optimizer_params = [
            {"params": [p for p in self.engine.backbone.parameters() if p.requires_grad], "lr": self.config.backbone_lr},
            {"params": self.engine.scorer.parameters(), "lr": self.config.head_lr}
        ]
        if self.engine.span_locator is not None:
            optimizer_params.append(
                {"params": self.engine.span_locator.parameters(), "lr": self.config.head_lr}
            )

        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=self.config.weight_decay)
        criterion_decision = RLCDCalibrationLoss(
            lambda_brier=self.config.lambda_brier,
            lambda_ece=self.config.lambda_ece
        )
        criterion_span = nn.BCEWithLogitsLoss()

        logger.info(f"Starting RedactX training for {num_epochs} epochs (Train: {len(train_records)}, Val: {len(val_records)})")
        print(f"Starting RedactX training for {num_epochs} epochs (Train: {len(train_records)}, Val: {len(val_records)})")

        history = []

        for epoch in range(num_epochs):
            self.engine.train()
            running_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.engine(input_ids, attention_mask)
                loss_dec = criterion_decision(outputs["logits"], labels)

                total_loss = loss_dec
                if self.engine.span_locator is not None and "span_logits" in outputs and "token_span_labels" in batch:
                    token_span_labels = batch["token_span_labels"].to(self.device)
                    loss_span = criterion_span(outputs["span_logits"], token_span_labels)
                    total_loss = loss_dec + 0.3 * loss_span

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=self.config.max_grad_norm)
                optimizer.step()

                running_loss += total_loss.item()

            epoch_train_loss = running_loss / max(len(train_loader), 1)

            # Validation Phase
            val_metrics = self.evaluate(val_loader, criterion_decision)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1"]
            val_recall = val_metrics["recall"]

            log_str = (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Recall (Sensitivity): {val_recall:.4f}"
            )
            print(log_str)
            logger.info(log_str)
            history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "val_loss": val_loss,
                **val_metrics
            })

        # Post-Training Platt Temperature Calibration
        print("Calibrating decision probabilities via Platt Temperature Scaling...")
        calib_temp = self.calibrate_temperature(val_loader)
        print(f"Optimal Platt calibration temperature: T = {calib_temp:.4f}")

        # Attach scores & metrics to engine
        scores_payload = {
            "final_metrics": history[-1] if history else {},
            "calibrated_temperature": calib_temp,
            "epochs": num_epochs,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "history": history
        }
        self.engine.scores = scores_payload

        # Persist Checkpoint Artifacts (weights, scores, config)
        self.engine.save_pretrained(save_dir)

        return {
            "history": history,
            "calibrated_temperature": calib_temp,
            "final_metrics": history[-1] if history else {},
            "checkpoint_dir": save_dir
        }

    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Runs evaluation over validation split, collecting metrics."""
        self.engine.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.engine(input_ids, attention_mask)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        avg_loss = total_loss / max(len(val_loader), 1)
        acc = accuracy_score(all_labels, all_preds) * 100.0
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.5

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc
        }

    def calibrate_temperature(self, val_loader: DataLoader) -> float:
        """Collects validation logits and fits Platt temperature scalar."""
        self.engine.eval()
        val_logits_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.engine(input_ids, attention_mask)
                val_logits_list.append(outputs["logits"].cpu())
                val_labels_list.append(labels.cpu())

        val_logits = torch.cat(val_logits_list, dim=0).to(self.device)
        val_labels = torch.cat(val_labels_list, dim=0).to(self.device)

        temp = self.engine.calibrator.fit(val_logits, val_labels)
        return temp


def train_redactx_engine(
    model: RedactXDecisionEngine,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    output_dir: str = "./redactx_checkpoint"
) -> Dict[str, Any]:
    """
    Direct function wrapper matching user's prototype signature.
    """
    cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        backbone_lr=learning_rate,
        head_lr=learning_rate * 5,
        output_dir=output_dir
    )
    trainer = RedactXTrainer(engine=model, config=cfg)
    return trainer.train(train_records, val_records)
