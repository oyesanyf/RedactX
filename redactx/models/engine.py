"""
RedactX Decision Engine: Differentially Private, Sub-50ms Jev-Like Decision Engine.
Binds Transformer backbone trunk with LoRA, single-pass Hobson pointer scoring head,
Platt temperature calibration, and token-level span attribution.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from redactx.config import RedactXConfig
from redactx.primitives import NoulDecision, DetectedSpan
from redactx.models.scorer import RedactXPointerScorer, PlattTemperatureScaler
from redactx.models.span_locator import TokenSpanLocator
from redactx.models.backbone import load_backbone_and_tokenizer

logger = logging.getLogger("redactx.engine")


class RedactXDecisionEngine(nn.Module):
    """
    RedactX: Differentially Private Decision Engine for Line-Rate PHI/PII Detection.
    System-1 non-generative decision model producing calibrated Jev primitives.
    """

    def __init__(self, config: Optional[RedactXConfig] = None):
        super().__init__()
        self.config = config or RedactXConfig()

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing RedactX Engine on device: {self.device}")

        # Load Backbone & Tokenizer
        self.backbone, self.tokenizer, hidden_dim, self.resolved_model_id = load_backbone_and_tokenizer(
            self.config, self.device
        )
        self.hidden_dim = hidden_dim

        # Hobson Pointer Scoring Head
        self.scorer = RedactXPointerScorer(
            hidden_dim=self.hidden_dim,
            num_classes=self.config.num_classes
        ).to(self.device)

        # Platt Temperature Scaler (Probability Calibration)
        self.calibrator = PlattTemperatureScaler(
            initial_temperature=self.config.temperature
        ).to(self.device)

        # Token-Level Span Attribution Head
        if self.config.enable_span_locator:
            self.span_locator = TokenSpanLocator(
                hidden_dim=self.hidden_dim
            ).to(self.device)
        else:
            self.span_locator = None

        # Benchmark / validation scores
        self.scores: Dict[str, Any] = {}

        self.to(self.device)

    def extract_context_embedding(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts pooled representation from terminal token of each sequence.
        """
        # Determine last non-padding token for each item in batch
        seq_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = seq_lengths.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lengths]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single-pass forward pass returning both decision logits and span logits.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None and isinstance(outputs, tuple):
            hidden_states = outputs[0]

        pooled = self.extract_context_embedding(hidden_states, attention_mask)
        decision_logits = self.scorer(pooled)

        res = {"logits": decision_logits, "hidden_states": hidden_states}

        if self.span_locator is not None:
            span_logits = self.span_locator(hidden_states)
            res["span_logits"] = span_logits

        return res

    @torch.inference_mode()
    def evaluate(
        self,
        text: str,
        query: str = "Does this text contain protected health information or personal identifiers?",
        decision_threshold: Optional[float] = None,
        extract_spans: Optional[bool] = None,
        payload_id: Optional[str] = None
    ) -> NoulDecision:
        """
        Executes single-pass scoring on arbitrary context and measures execution latency.
        Returns a calibrated Jev 'noul' primitive with optional detected spans.
        """
        threshold = decision_threshold if decision_threshold is not None else self.config.decision_threshold
        locate_spans = extract_spans if extract_spans is not None else self.config.enable_span_locator

        prompt = f"Context: {text}\nQuery: {query}\nDecision:"

        t_start = time.perf_counter()

        # Tokenize with offsets
        encoding = self.tokenizer(
            prompt,
            max_length=self.config.max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        offset_mapping = encoding.offset_mapping[0].tolist()

        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        raw_logits = outputs["logits"]

        # Platt Temperature Scaled Logits
        scaled_logits = self.calibrator(raw_logits)
        probabilities = torch.softmax(scaled_logits, dim=-1).squeeze(0).tolist()

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        phi_prob = round(float(probabilities[1]), 4)
        clean_prob = round(float(probabilities[0]), 4)
        verdict = "Contains_PHI_PII" if phi_prob >= threshold else "Clean"
        passed_gate = (verdict == "Clean")

        # Extract Spans if flagged or requested
        detected_spans: List[DetectedSpan] = []
        if locate_spans and self.span_locator is not None and "span_logits" in outputs:
            span_probs = torch.sigmoid(outputs["span_logits"]).squeeze(0)
            # If model found PHI or text has notable probability, locate spans
            if phi_prob >= 0.20:
                detected_spans = self.span_locator.extract_spans(
                    token_probs=span_probs,
                    offset_mapping=offset_mapping,
                    full_prompt=prompt,
                    context_text=text,
                    threshold=0.40
                )

        return NoulDecision(
            id=payload_id,
            primitive="noul",
            context=text,
            query=query,
            verdict=verdict,
            phi_probability=phi_prob,
            clean_probability=clean_prob,
            passed_gate=passed_gate,
            calibrated=True,
            latency_ms=round(t_elapsed_ms, 2),
            spans=detected_spans,
            metadata={
                "base_model": self.resolved_model_id,
                "temperature": round(float(self.calibrator.temperature.item()), 3),
                "threshold": threshold,
                "device": str(self.device)
            }
        )

    @torch.inference_mode()
    def evaluate_batch(
        self,
        texts: List[str],
        query: str = "Does this text contain protected health information or personal identifiers?",
        decision_threshold: Optional[float] = None
    ) -> List[NoulDecision]:
        """
        Executes high-throughput batch evaluation.
        """
        threshold = decision_threshold if decision_threshold is not None else self.config.decision_threshold
        prompts = [f"Context: {t}\nQuery: {query}\nDecision:" for t in texts]

        t_start = time.perf_counter()
        encodings = self.tokenizer(
            prompts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        self.eval()
        outputs = self.forward(input_ids, attention_mask)
        scaled_logits = self.calibrator(outputs["logits"])
        probabilities = torch.softmax(scaled_logits, dim=-1).tolist()
        batch_latency = (time.perf_counter() - t_start) * 1000.0
        avg_latency = batch_latency / max(len(texts), 1)

        decisions = []
        for idx, text in enumerate(texts):
            probs = probabilities[idx]
            phi_prob = round(float(probs[1]), 4)
            clean_prob = round(float(probs[0]), 4)
            verdict = "Contains_PHI_PII" if phi_prob >= threshold else "Clean"
            decisions.append(NoulDecision(
                id=f"batch_{idx}",
                primitive="noul",
                context=text,
                query=query,
                verdict=verdict,
                phi_probability=phi_prob,
                clean_probability=clean_prob,
                passed_gate=(verdict == "Clean"),
                calibrated=True,
                latency_ms=round(avg_latency, 2),
                spans=[]
            ))
        return decisions

    def save_checkpoint(self, output_dir: str = "./models/redactx"):
        """Alias for save_pretrained."""
        self.save_pretrained(output_dir)

    def save_pretrained(self, output_dir: str = "./models/redactx"):
        """
        Persists full model package to the models folder:
        - Meta configuration (redactx_meta.json)
        - LoRA adapter weights
        - Hobson pointer scorer weights (redactx_scorer.pt)
        - Platt temperature calibrator (redactx_calibrator.pt)
        - Token span locator (redactx_span_locator.pt)
        - Tokenizer files
        """
        os.makedirs(output_dir, exist_ok=True)

        meta = {
            "base_model_id": self.config.base_model_id,
            "resolved_model_id": self.resolved_model_id,
            "num_classes": self.config.num_classes,
            "hidden_dim": self.hidden_dim,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "max_length": self.config.max_length,
            "decision_threshold": self.config.decision_threshold,
            "temperature": float(self.calibrator.temperature.item()),
            "enable_span_locator": self.span_locator is not None
        }

        with open(os.path.join(output_dir, "redactx_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save scores file (validation accuracy, f1, recall, AUC, calibrated T, loss history)
        if self.scores:
            with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
                json.dump(self.scores, f, indent=2)

        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(os.path.join(output_dir, "lora_adapter"))

        torch.save(self.scorer.state_dict(), os.path.join(output_dir, "redactx_scorer.pt"))
        torch.save(self.calibrator.state_dict(), os.path.join(output_dir, "redactx_calibrator.pt"))

        if self.span_locator is not None:
            torch.save(self.span_locator.state_dict(), os.path.join(output_dir, "redactx_span_locator.pt"))

        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"RedactX model successfully exported to: {output_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Loads weights and scores into scorer, calibrator, span locator, and backbone LoRA adapter."""
        scorer_path = os.path.join(checkpoint_dir, "redactx_scorer.pt")
        if os.path.exists(scorer_path):
            self.scorer.load_state_dict(torch.load(scorer_path, map_location=self.device))
            logger.info("Loaded Hobson scorer weights.")

        calib_path = os.path.join(checkpoint_dir, "redactx_calibrator.pt")
        if os.path.exists(calib_path):
            self.calibrator.load_state_dict(torch.load(calib_path, map_location=self.device))
            logger.info("Loaded Platt calibrator weights.")

        span_path = os.path.join(checkpoint_dir, "redactx_span_locator.pt")
        if os.path.exists(span_path) and self.span_locator is not None:
            self.span_locator.load_state_dict(torch.load(span_path, map_location=self.device))
            logger.info("Loaded span locator weights.")

        # Load scores
        scores_path = os.path.join(checkpoint_dir, "scores.json")
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                self.scores = json.load(f)
            logger.info(f"Loaded model scores: {self.scores.get('final_metrics', {})}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./models/redactx",
        device: Optional[str] = None
    ) -> "RedactXDecisionEngine":
        """
        Loads a trained RedactX decision model from the models folder in a single line of code:
            model = RedactX.from_pretrained("models/redactx")
        """
        meta_file = os.path.join(model_path, "redactx_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            config = RedactXConfig(
                base_model_id=meta.get("resolved_model_id", meta.get("base_model_id")),
                num_classes=meta.get("num_classes", 2),
                hidden_dim=meta.get("hidden_dim"),
                lora_r=meta.get("lora_r", 16),
                lora_alpha=meta.get("lora_alpha", 32),
                max_length=meta.get("max_length", 256),
                decision_threshold=meta.get("decision_threshold", 0.50),
                temperature=meta.get("temperature", 1.0),
                enable_span_locator=meta.get("enable_span_locator", True),
                device=device
            )
        else:
            config = RedactXConfig(device=device)

        instance = cls(config=config)
        instance.load_checkpoint(model_path)
        return instance


# Convenient alias matching user's original class name
RedactX = RedactXDecisionEngine

