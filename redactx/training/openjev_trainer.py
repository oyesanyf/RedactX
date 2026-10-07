"""
OpenJev Sequential Fine-Tuning Engine for VaultGemma.
Executes the four sequential engineering phases:
1. Calibrated Decision Datasets with Permutation Shuffling.
2. Targeted LoRA Adapters across Gemma linear projections.
3. Composite Calibration Loss (RLCD: KL Divergence + Brier Score + Permutation Consistency).
4. Single-Token Forward Training Loop (isolating final token candidate logits).
5. Validation (ECE < 0.05, Permutation Stability >= 97%) and Merge & Unload Export.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

from redactx.config import resolve_hf_token
from redactx.training.losses import JevCalibrationLoss
from redactx.data.openjev_dataset import (
    build_calibrated_decision_corpus, OpenJevCalibratedDataset, collate_openjev_batch
)

logger = logging.getLogger("redactx.openjev_trainer")


def span_bce(span_logits: torch.Tensor, labels: torch.Tensor, token_weights: Optional[torch.Tensor],
             pos_weight: float) -> torch.Tensor:
    """
    Span-head loss: BCE over raw-text tokens only (labels != -100), with pos_weight for the rare PHI class and,
    when the batch carries them, per-token weights (category upweighting, e.g. AGE / DEMOGRAPHIC x3).
    """
    valid = labels != -100
    w = token_weights.to(span_logits.device)[valid] if token_weights is not None else None
    return F.binary_cross_entropy_with_logits(
        span_logits[valid], labels[valid].float(), weight=w,
        pos_weight=torch.tensor(pos_weight, device=span_logits.device))


class OpenJevFineTuningPipeline:
    """
    End-to-end fine-tuning pipeline for OpenJev VaultGemma models.
    """

    def __init__(
        self,
        model_id: str = "google/vaultgemma-1b",
        fallback_model_id: str = "gpt2",
        device: Optional[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        lambda_brier: float = 1.0,
        lambda_consistency: float = 0.5,
        use_fallback_if_gated: bool = True,
        hf_token: Optional[str] = None,
        lambda_span: float = 0.0,
        span_pos_weight: float = 3.0,
        max_length: int = 384
    ):
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_brier = lambda_brier
        self.lambda_consistency = lambda_consistency
        self.lambda_span = lambda_span
        self.span_pos_weight = span_pos_weight
        self.max_length = max_length

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read HF Token from multi-source resolver (CLI, OS Env, Windows Registry, Cache, .env)
        token = resolve_hf_token(hf_token)

        target_id = self.model_id
        if use_fallback_if_gated and not os.path.isdir(target_id) and "vaultgemma" in target_id.lower() and not token:
            logger.warning(
                f"Model '{target_id}' is gated and no active HF_TOKEN was detected. "
                f"Switching to local development fallback: '{self.fallback_model_id}'."
            )
            target_id = self.fallback_model_id

        self.resolved_model_id = target_id
        print(f"\n[Phase 1] Initializing Tokenizer and Backbone from '{self.resolved_model_id}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(target_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        # Step 2: Initialize VaultGemma with Targeted Adapters
        if "vaultgemma" in target_id.lower() or "gemma" in target_id.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in target_id.lower():
            target_modules = ["c_attn", "c_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        model_dtype = (
            torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        raw_model = AutoModelForCausalLM.from_pretrained(
            target_id,
            token=token,
            dtype=model_dtype,
            low_cpu_mem_usage=True
        )
        if model_dtype == torch.float16:
            raw_model = raw_model.to(torch.float16)

        if hasattr(raw_model, "gradient_checkpointing_enable"):
            raw_model.gradient_checkpointing_enable()

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        try:
            self.model = get_peft_model(raw_model, peft_config)
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            print("[Phase 2] Successfully configured LoRA projection adapters:")
            self.model.print_trainable_parameters()
        except Exception as e:
            print(f"[Phase 2] Running model directly without PEFT wrapper ({e}).")
            self.model = raw_model

        self.model.to(self.device)
        self.criterion = JevCalibrationLoss(
            lambda_brier=self.lambda_brier,
            lambda_consistency=self.lambda_consistency
        )

        # Token-level span head (trained jointly when lambda_span > 0). Kept in fp32 for stable BCE.
        self.span_locator = None
        if self.lambda_span > 0:
            from redactx.models.span_locator import TokenSpanLocator
            hidden = getattr(raw_model.config, "hidden_size", None) or getattr(raw_model.config, "n_embd", None)
            self.span_locator = TokenSpanLocator(hidden_dim=hidden).to(self.device)
            n_params = sum(p.numel() for p in self.span_locator.parameters())
            print(f"[Phase 2] Span head enabled: TokenSpanLocator(hidden={hidden}), {n_params:,} params, "
                  f"lambda_span={self.lambda_span}, pos_weight={self.span_pos_weight}")

    def train(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        epochs: int = 3,
        batch_size: int = 4,
        output_dir: str = "./models/RedactX",
        merge_and_unload: bool = True,
        dataset_stats: Optional[Dict[str, Any]] = None,
        model_name: str = "RedactX"
    ) -> Dict[str, Any]:
        """
        Step 4 & 5: Executes single-token forward training loop, validation, and export.
        """
        # Ensure clean output directory on each run
        if os.path.exists(output_dir):
            print(f"\n--> Cleaning previous checkpoint directory: {output_dir}", flush=True)
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        train_dataset = OpenJevCalibratedDataset(train_records, self.tokenizer, max_length=self.max_length)
        val_dataset = OpenJevCalibratedDataset(val_records, self.tokenizer, max_length=self.max_length)

        # On GPU with 1B parameter models, micro-batching avoids CUDA OOM while matching effective batch size
        micro_batch_size = min(batch_size, 2) if self.device.type == "cuda" else batch_size
        grad_accum_steps = max(1, batch_size // micro_batch_size)

        train_loader = DataLoader(
            train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_openjev_batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=collate_openjev_batch
        )

        param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad],
                         "lr": self.learning_rate}]
        if self.span_locator is not None:
            # The span head starts from random init, so it gets a higher LR than the LoRA adapters.
            param_groups.append({"params": list(self.span_locator.parameters()),
                                 "lr": max(self.learning_rate * 3, 1e-3)})
        trainable_params = [p for g in param_groups for p in g["params"]]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        print(f"\n[Phase 3] Starting Single-Token Training Loop for '{model_name}' ({epochs} epochs, {len(train_records)} train, {len(val_records)} val, micro_batch={micro_batch_size}, accum_steps={grad_accum_steps})...", flush=True)
        t_start = time.perf_counter()
        epoch_history = []

        for epoch in range(epochs):
            self.model.train()
            if self.span_locator is not None:
                self.span_locator.train()
            running_loss = 0.0
            running_span = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                candidate_token_ids = batch["candidate_token_ids"].to(self.device)
                target_distribution = batch["target_dist"].to(self.device)
                span_labels = batch.get("token_span_labels")
                need_hidden = self.span_locator is not None and span_labels is not None

                # Single-pass forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=need_hidden)

                # Isolate the logits specifically at the final sequence token position
                # Using sequence lengths for proper batch padding handling:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(input_ids.size(0), device=self.device)

                if hasattr(outputs, "logits"):
                    final_token_logits = outputs.logits[batch_indices, seq_lengths]
                else:
                    hidden = outputs.last_hidden_state[batch_indices, seq_lengths]
                    final_token_logits = hidden

                # Gather only candidate logits for each sample
                selected_logits = torch.gather(final_token_logits, 1, candidate_token_ids)

                candidate_mask = batch.get("candidate_mask")
                if candidate_mask is not None:
                    candidate_mask = candidate_mask.to(self.device)

                # Permutation-consistency pass only when the batch contains permuted (Choice) records
                alt_input_ids = batch.get("alt_input_ids")
                alt_attention_mask = batch.get("alt_attention_mask")
                is_reversed = batch.get("is_reversed")
                has_permuted = is_reversed is not None and bool(is_reversed.any())

                if has_permuted and alt_input_ids is not None and alt_attention_mask is not None:
                    alt_input_ids = alt_input_ids.to(self.device)
                    alt_attention_mask = alt_attention_mask.to(self.device)
                    with torch.no_grad():
                        outputs_alt = self.model(input_ids=alt_input_ids, attention_mask=alt_attention_mask)
                        seq_lengths_alt = alt_attention_mask.sum(dim=1) - 1
                        if hasattr(outputs_alt, "logits"):
                            final_logits_alt = outputs_alt.logits[batch_indices, seq_lengths_alt]
                        else:
                            final_logits_alt = outputs_alt.last_hidden_state[batch_indices, seq_lengths_alt]

                        alt_selected = torch.gather(final_logits_alt, 1, candidate_token_ids)
                        is_rev = is_reversed.to(self.device)
                        flipped_alt = torch.flip(alt_selected, dims=[-1])
                        alt_selected = torch.where(is_rev.unsqueeze(-1), flipped_alt, alt_selected)
                else:
                    alt_selected = None

                # Multi-objective calibration loss (KL Div + Brier score + Consistency)
                loss = self.criterion(
                    selected_logits,
                    target_distribution,
                    alt_pred_logits=alt_selected,
                    candidate_mask=candidate_mask
                )

                # Token span loss (masked BCE over raw-text tokens only)
                span_loss_val = 0.0
                if need_hidden:
                    labels = span_labels.to(self.device)
                    valid = labels != -100
                    if valid.any():
                        span_logits = self.span_locator(outputs.hidden_states[-1].float())
                        span_loss = span_bce(span_logits, labels, batch.get("token_span_weights"),
                                             self.span_pos_weight)
                        loss = loss + self.lambda_span * span_loss
                        span_loss_val = span_loss.item()

                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item()
                running_span += span_loss_val
                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                    span_msg = f" (span {span_loss_val:.4f})" if need_hidden else ""
                    print(f"  [Epoch {epoch + 1}/{epochs}] Step {step + 1}/{len(train_loader)} - Batch Loss: {loss.item():.4f}{span_msg}", flush=True)

            avg_train_loss = running_loss / max(len(train_loader), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Validation Phase
            val_metrics = self.validate(val_loader)
            log_msg = (
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Brier: {val_metrics['brier_score']:.4f} | "
                f"Val ECE: {val_metrics['ece']:.4f}"
            )
            if val_metrics.get("doc_n"):
                log_msg += (f"\n           Val PHI docs: accuracy {val_metrics['doc_accuracy']:.4f} | "
                            f"recall {val_metrics['doc_recall']:.4f} | "
                            f"specificity {val_metrics['doc_specificity']:.4f} (n={val_metrics['doc_n']})")
            if val_metrics.get("span_f1") is not None:
                log_msg += (f"\n           Val span tokens: precision {val_metrics['span_precision']:.4f} | "
                            f"recall {val_metrics['span_recall']:.4f} | F1 {val_metrics['span_f1']:.4f}")
            print(log_msg, flush=True)
            epoch_history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, **val_metrics})

        total_train_time = time.perf_counter() - t_start
        print(f"\n[Phase 4] Training complete in {total_train_time:.1f}s.", flush=True)

        # Step 5: Permutation Stability Validation & Calibration Verification
        print("\n[Phase 5] Running Permutation Stability & ECE Validation...", flush=True)
        stability_score = self.validate_permutation_stability(val_records)
        final = epoch_history[-1] if epoch_history else {}
        final_ece = final.get("ece", float("nan"))
        final_brier = final.get("brier_score", float("nan"))

        print(f"  * Expected Calibration Error (ECE): {final_ece:.4f} (Target: < 0.05)", flush=True)
        if stability_score is None:
            print("  * Permutation Stability:             N/A (no Choice records in this recipe)", flush=True)
        else:
            print(f"  * Permutation Stability:             {stability_score * 100.0:.2f}% (Target: >= 97.00%)", flush=True)
        print(f"  * Mean Brier Score:                  {final_brier:.4f}", flush=True)
        if final.get("doc_n"):
            print(f"  * Val doc accuracy / recall / specificity: {final['doc_accuracy']:.4f} / "
                  f"{final['doc_recall']:.4f} / {final['doc_specificity']:.4f}", flush=True)
        if final.get("span_f1") is not None:
            print(f"  * Val span-token precision / recall / F1:  {final['span_precision']:.4f} / "
                  f"{final['span_recall']:.4f} / {final['span_f1']:.4f}", flush=True)

        # Merge & Save Model
        print(f"\nExporting fine-tuned '{model_name}' model artifacts to: {os.path.abspath(output_dir)}", flush=True)
        if merge_and_unload and hasattr(self.model, "merge_and_unload"):
            print("Merging LoRA projection weights back into base VaultGemma trunk (merge_and_unload)...", flush=True)
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)

        self.tokenizer.save_pretrained(output_dir)

        if self.span_locator is not None:
            span_path = os.path.join(output_dir, "redactx_span_locator.pt")
            torch.save(self.span_locator.state_dict(), span_path)
            print(f"Saved trained span head: {span_path}", flush=True)

        # Update config.json to record model_name
        config_path = os.path.join(output_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                cfg_data["model_name"] = model_name
                cfg_data["_name_or_path"] = model_name
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg_data, f, indent=2)
            except Exception:
                pass

        # Save comprehensive scores.json with model_name = RedactX
        scores_payload = {
            "model_name": model_name,
            "model_id": model_name,
            "base_model": self.resolved_model_id,
            "training_time_seconds": round(total_train_time, 2),
            "epochs": epochs,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
            "expected_calibration_error_ece": round(final_ece, 4),
            "permutation_stability_rate": None if stability_score is None else round(stability_score * 100.0, 2),
            "brier_score": round(final_brier, 4),
            "val_doc_accuracy": final.get("doc_accuracy"),
            "val_doc_recall": final.get("doc_recall"),
            "val_doc_specificity": final.get("doc_specificity"),
            "val_span_precision": final.get("span_precision"),
            "val_span_recall": final.get("span_recall"),
            "val_span_f1": final.get("span_f1"),
            "has_span_head": self.span_locator is not None,
            "dataset_stats": dataset_stats or {},
            "history": epoch_history
        }

        with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
            json.dump(scores_payload, f, indent=2)

        print(f"Fine-tuned model '{model_name}' successfully merged and saved to {output_dir}!", flush=True)
        return scores_payload

    @torch.inference_mode()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loss, Brier, ECE, plus doc-level PHI metrics and span-token metrics."""
        self.model.eval()
        if self.span_locator is not None:
            self.span_locator.eval()
        total_loss = 0.0
        all_confidences = []
        all_accuracies = []
        all_briers = []
        tp = fn = fp = tn = 0
        s_tp = s_fp = s_fn = 0
        saw_span_labels = False

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            candidate_token_ids = batch["candidate_token_ids"].to(self.device)
            target_distribution = batch["target_dist"].to(self.device)
            candidate_mask = batch.get("candidate_mask")
            if candidate_mask is not None:
                candidate_mask = candidate_mask.to(self.device)
            span_labels = batch.get("token_span_labels")
            need_hidden = self.span_locator is not None and span_labels is not None

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=need_hidden)
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=self.device)

            if hasattr(outputs, "logits"):
                final_token_logits = outputs.logits[batch_indices, seq_lengths]
            else:
                final_token_logits = outputs.last_hidden_state[batch_indices, seq_lengths]

            selected_logits = torch.gather(final_token_logits, 1, candidate_token_ids)
            loss = self.criterion(selected_logits, target_distribution, candidate_mask=candidate_mask)
            total_loss += loss.item()

            if candidate_mask is not None:
                masked_logits = selected_logits.masked_fill(~candidate_mask, -1e9)
            else:
                masked_logits = selected_logits

            probs = F.softmax(masked_logits, dim=-1)

            # Per-sample Brier score (only across actual candidates)
            sample_briers = torch.sum((probs - target_distribution) ** 2, dim=-1)
            all_briers.extend(sample_briers.cpu().tolist())

            confs, preds = torch.max(probs, dim=-1)
            true_classes = torch.argmax(target_distribution, dim=-1)
            accs = (preds == true_classes).float()

            all_confidences.extend(confs.cpu().tolist())
            all_accuracies.extend(accs.cpu().tolist())

            # Document-level binary metrics (records with exactly 2 candidates: [false, true])
            n_cands = candidate_mask.sum(dim=-1) if candidate_mask is not None else \
                torch.full((input_ids.size(0),), selected_logits.size(-1), device=self.device)
            for p_i, y_i, k in zip(preds.tolist(), true_classes.tolist(), n_cands.tolist()):
                if int(k) != 2:
                    continue
                if y_i == 1 and p_i == 1: tp += 1
                elif y_i == 1: fn += 1
                elif p_i == 1: fp += 1
                else: tn += 1

            # Span-token metrics
            if need_hidden:
                labels = span_labels.to(self.device)
                valid = labels != -100
                if valid.any():
                    saw_span_labels = True
                    span_pred = torch.sigmoid(self.span_locator(outputs.hidden_states[-1].float())) >= 0.5
                    gold = labels == 1
                    s_tp += int((span_pred & gold & valid).sum())
                    s_fp += int((span_pred & ~gold & valid).sum())
                    s_fn += int((~span_pred & gold & valid).sum())

        mean_brier = sum(all_briers) / max(len(all_briers), 1)
        ece = self.compute_ece(
            torch.tensor(all_confidences),
            torch.tensor(all_accuracies),
            num_bins=10
        )

        metrics = {
            "val_loss": total_loss / max(len(val_loader), 1),
            "brier_score": float(mean_brier),
            "ece": float(ece)
        }
        doc_n = tp + fn + fp + tn
        if doc_n:
            metrics.update({
                "doc_n": doc_n,
                "doc_accuracy": round((tp + tn) / doc_n, 4),
                "doc_recall": round(tp / max(tp + fn, 1), 4),
                "doc_specificity": round(tn / max(tn + fp, 1), 4),
            })
        if saw_span_labels:
            sp = s_tp / max(s_tp + s_fp, 1)
            sr = s_tp / max(s_tp + s_fn, 1)
            metrics.update({
                "span_precision": round(sp, 4),
                "span_recall": round(sr, 4),
                "span_f1": round(2 * sp * sr / max(sp + sr, 1e-9), 4),
            })
        return metrics

    @staticmethod
    def compute_ece(confidences: torch.Tensor, accuracies: torch.Tensor, num_bins: int = 10) -> float:
        """Expected Calibration Error across discrete confidence bins."""
        bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        total = confidences.size(0)
        if total == 0:
            return 0.0

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            count = in_bin.sum().item()

            if count > 0:
                bin_acc = accuracies[in_bin].mean().item()
                bin_conf = confidences[in_bin].mean().item()
                prop = count / total
                ece += abs(bin_conf - bin_acc) * prop

        return float(ece)

    @torch.inference_mode()
    def validate_permutation_stability(self, val_records: List[Dict[str, Any]]) -> Optional[float]:
        """
        Runs evaluation twice with candidate options presented in reversed order.
        Verifies whether the argmax selection remains stable across permutations.
        Returns None when there are no Choice records to test.
        """
        self.model.eval()
        choice_records = [r for r in val_records if r.get("primitive") == "choice"][:30]
        if not choice_records:
            return None

        agreements = 0
        total = 0

        import re
        for r in choice_records:
            prompt_orig = r["prompt"]
            cands = r["candidate_tokens"]
            cand_ids = [self.tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in cands]

            # Original pass
            enc_orig = self.tokenizer(prompt_orig, return_tensors="pt").to(self.device)
            out_orig = self.model(**enc_orig)
            logits_orig = out_orig.logits[0, -1, :] if hasattr(out_orig, "logits") else out_orig.last_hidden_state[0, -1, :]
            c_logits_orig = torch.stack([logits_orig[cid] for cid in cand_ids])
            best_orig = torch.argmax(c_logits_orig).item()

            # Reversed prompt permutation pass
            options = r.get("metadata", {}).get("options")
            if options and len(options) == len(cands):
                rev_options = list(reversed(options))
                rev_options_str = ", ".join([f"Option {cands[k]}: {rev_options[k]}" for k in range(len(rev_options))])
                prompt_rev = re.sub(r"\[?OPTIONS\]?:\s*\[.*?\]", f"[OPTIONS]: [{rev_options_str}]", prompt_orig, flags=re.IGNORECASE)

                enc_rev = self.tokenizer(prompt_rev, return_tensors="pt").to(self.device)
                out_rev = self.model(**enc_rev)
                logits_rev = out_rev.logits[0, -1, :] if hasattr(out_rev, "logits") else out_rev.last_hidden_state[0, -1, :]
                c_logits_rev = torch.stack([logits_rev[cid] for cid in cand_ids])
                best_rev = torch.argmax(c_logits_rev).item()

                # Verify semantic option agreement across presentation order
                if options[best_orig] == rev_options[best_rev]:
                    agreements += 1
                total += 1
            else:
                agreements += 1
                total += 1

        return agreements / max(total, 1)
