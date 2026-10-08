"""
OpenJev Decision Engine for VaultGemma.
Implements the OpenJev non-generative architecture for autoregressive transformers:
1. Prefix conditioning with target logit projections (bypassing autoregressive loop).
2. The three Jev primitives: Noul (binary hypothesis), Choice (constrained categorical),
   and Score (ordinal distribution & mathematical expectation).
3. Active re-reads and evidential calibration via Shannon entropy thresholding and MC perturbation.
4. Wire-compatible with TypeSafe Jev & OpenJev specifications.
"""

import os
import json
import math
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel

from redactx.config import RedactXConfig
from redactx.primitives import (
    QuestionPayload, DecisionRequest, DecisionResponse,
    NoulResult, ChoiceResult, ScoreResult, DetectedSpan, NoulDecision
)
from redactx.models.span_locator import TokenSpanLocator
from redactx.data.prompting import build_noul_prompt, token_raw_ranges

logger = logging.getLogger("redactx.openjev")


def resolve_single_token(tokenizer: AutoTokenizer, token_str: str) -> int:
    """Safely extracts a single vocabulary token ID with prefix whitespace awareness."""
    clean_str = token_str.strip()
    # Try with leading whitespace first (standard for sentencepiece/BPE word starts)
    ids_with_space = tokenizer.encode(f" {clean_str}", add_special_tokens=False)
    if ids_with_space:
        return ids_with_space[-1]
    # Fallback to direct token encoding
    ids = tokenizer.encode(clean_str, add_special_tokens=False)
    return ids[-1]


class OpenJevVaultGemmaEngine(nn.Module):
    """
    OpenJev Decision Engine wrapped around Google's VaultGemma 1B model.
    Bypasses autoregressive text generation to achieve sub-50ms line-rate decisions.
    """

    def __init__(self, config: Optional[RedactXConfig] = None):
        super().__init__()
        self.config = config or RedactXConfig()

        if self.config.device is not None:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read HF Token from environment
        token = os.getenv("HF_TOKEN")
        if not token:
            try:
                from huggingface_hub import get_token
                token = get_token()
            except Exception:
                token = None

        target_model_id = self.config.base_model_id
        if self.config.use_fallback_if_gated and "vaultgemma" in target_model_id.lower() and not token:
            logger.warning(
                f"Model '{target_model_id}' is gated and no active HF_TOKEN was detected. "
                f"Switching to local development fallback: '{self.config.fallback_model_id}'."
            )
            target_model_id = self.config.fallback_model_id

        logger.info(f"Loading OpenJev backbone from '{target_model_id}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        # Check model configuration
        is_causal = False
        try:
            m_config = AutoConfig.from_pretrained(target_model_id, token=token)
            architectures = getattr(m_config, "architectures", None) or []
            model_type = str(getattr(m_config, "model_type", "")).lower()
            if (any(("CausalLM" in a) or ("LMHeadModel" in a) for a in architectures)
                    or "gemma" in target_model_id.lower() or model_type.startswith(("gemma", "vaultgemma", "gpt2"))):
                is_causal = True
        except Exception:
            pass

        self.is_causal = is_causal
        is_local_dir = os.path.isdir(target_model_id)

        if is_causal:
            raw_model = AutoModelForCausalLM.from_pretrained(
                target_model_id,
                token=token,
                dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            hidden_dim = getattr(raw_model.config, "hidden_size", getattr(raw_model.config, "n_embd", 2048))
            if "gpt2" in target_model_id.lower():
                target_modules = ["c_attn", "c_proj"]
            elif "vaultgemma" in target_model_id.lower() or "gemma" in target_model_id.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                target_modules = ["q_proj", "v_proj"]
        else:
            from transformers import AutoModel
            raw_model = AutoModel.from_pretrained(target_model_id, token=token)
            hidden_dim = getattr(raw_model.config, "hidden_size", getattr(raw_model.config, "dim", 768))
            target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"] if "distilbert" in target_model_id.lower() else ["query", "value"]

        # Only attach fresh LoRA adapters if not loading an already merged model directory
        if not is_local_dir:
            try:
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none"
                )
                self.model = get_peft_model(raw_model, lora_config)
            except Exception as e:
                logger.info(f"Using backbone directly without PEFT adapter wrapper ({e})")
                self.model = raw_model
        else:
            self.model = raw_model

        self.hidden_dim = hidden_dim
        self.resolved_model_id = target_model_id

        # Resolve anchor token IDs in tokenizer vocabulary
        self.true_token_id = resolve_single_token(self.tokenizer, "true")
        self.false_token_id = resolve_single_token(self.tokenizer, "false")
        self.yes_token_id = resolve_single_token(self.tokenizer, "yes")
        self.no_token_id = resolve_single_token(self.tokenizer, "no")

        # Readout projection head for encoder fallbacks if causal LM head is missing
        if not hasattr(self.model, "lm_head") and not is_causal:
            self.readout_head = nn.Linear(hidden_dim, len(self.tokenizer)).to(self.device)
        else:
            self.readout_head = None

        # Token-Level Span Attribution Head for PHI localization (only if a TRAINED head was saved).
        # Kept in fp32, exactly as in training (head applied to hidden_states[-1].float()).
        # span_threshold / doc_threshold are set below from redactx_thresholds.json (defaults 0.5)
        span_file = os.path.join(target_model_id, "redactx_span_locator.pt") if is_local_dir else None
        if self.config.enable_span_locator and span_file and os.path.exists(span_file):
            self.span_locator = TokenSpanLocator(hidden_dim=hidden_dim).to(self.device)
            self.span_locator.load_state_dict(torch.load(span_file, map_location=self.device))
            self.span_locator.eval()
        else:
            self.span_locator = None

        # Calibration Temperature (Platt Scaling)
        self.temperature = 1.0
        if is_local_dir:
            temp_file = os.path.join(target_model_id, "temperature.json")
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        self.temperature = float(json.load(f).get("temperature", 1.0))
                except Exception:
                    self.temperature = 1.0

        # Operating thresholds chosen for a target recall on held-out data (calibrate_thresholds.py)
        from redactx.production.thresholds import ThresholdConfig
        self.thresholds = ThresholdConfig.load(target_model_id) if is_local_dir else ThresholdConfig()
        self.doc_threshold = float(self.thresholds.doc_threshold)
        self.span_threshold = float(self.thresholds.span_threshold)

        self.scores: Dict[str, Any] = {}
        self.to(self.device)
        self.model.eval()

    def extract_terminal_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Runs single-pass forward pass and extracts vocabulary logits at the final prompt token.
        [batch_size, vocab_size]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if hasattr(outputs, "logits") and outputs.logits is not None:
            # Autoregressive causal LM logits: [batch_size, seq_len, vocab_size]
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            return outputs.logits[batch_idx, seq_lens]
        else:
            # Encoder representation through readout head
            hidden_states = getattr(outputs, "last_hidden_state", outputs[0])
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            pooled = hidden_states[batch_idx, seq_lens]
            return self.readout_head(pooled)

    def candidate_logits_fp32(self, outputs, rows: torch.Tensor, last: torch.Tensor,
                              token_ids: List[int]) -> torch.Tensor:
        """
        Logits of `token_ids` at positions (rows, last), computed in float32 from the final hidden state and
        the LM-head rows. Under bf16 the full-vocabulary logits are rounded to bf16 (step 0.5 at |logit|~100),
        which visibly moves P(true); this keeps the decision precise. Falls back to outputs.logits when the
        model exposes no output embedding / hidden states. [len(rows), len(token_ids)]
        """
        head = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        hs = getattr(outputs, "hidden_states", None)
        if head is not None and hs is not None and getattr(head, "weight", None) is not None:
            h = hs[-1][rows, last].float()
            idx = torch.tensor(token_ids, device=head.weight.device)
            w = head.weight.index_select(0, idx).float()
            logits = h @ w.T
            bias = getattr(head, "bias", None)
            if bias is not None:
                logits = logits + bias.index_select(0, idx).float()
            cap = getattr(getattr(self.model, "config", None), "final_logit_softcapping", None)
            if cap:
                logits = torch.tanh(logits / cap) * cap
            return logits
        if getattr(outputs, "logits", None) is not None:
            return outputs.logits[rows, last][:, token_ids].float()
        return self.readout_head(outputs.last_hidden_state[rows, last])[:, token_ids].float()

    @staticmethod
    def calculate_entropy(probs: torch.Tensor) -> float:
        """Calculates normalized Shannon entropy H(P) in [0, 1]."""
        k = probs.size(-1)
        if k <= 1:
            return 0.0
        p = probs.clamp(min=1e-12)
        entropy = -torch.sum(p * torch.log2(p)).item()
        max_entropy = math.log2(k)
        return float(entropy / max_entropy)

    def _execute_reread_perturbation(
        self,
        prompt: str,
        target_token_ids: List[int],
        num_rereads: int = 3,
        noise_std: float = 0.02
    ) -> torch.Tensor:
        """
        Executes active re-reads with Monte Carlo embedding perturbation to stabilize uncertainty.
        """
        encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        accumulated_probs = []

        # Get base embedding layer
        embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None

        for _ in range(num_rereads):
            if embed_layer is not None:
                # Add slight Gaussian noise perturbation to input embeddings
                orig_embeds = embed_layer(encoding.input_ids)
                noise = torch.randn_like(orig_embeds) * noise_std
                perturbed_embeds = orig_embeds + noise
                outputs = self.model(inputs_embeds=perturbed_embeds, attention_mask=encoding.attention_mask)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits[0, -1, :]
                else:
                    logits = self.readout_head(outputs.last_hidden_state[0, -1, :])
            else:
                outputs = self.model(**encoding)
                logits = outputs.logits[0, -1, :] if hasattr(outputs, "logits") else self.readout_head(outputs.last_hidden_state[0, -1, :])

            candidate_logits = torch.stack([logits[tid] for tid in target_token_ids])
            probs = F.softmax(candidate_logits, dim=-1)
            accumulated_probs.append(probs)

        # Average distributions across re-reads
        stacked = torch.stack(accumulated_probs, dim=0)
        return torch.mean(stacked, dim=0)

    @torch.inference_mode()
    def evaluate_decision(self, req: DecisionRequest) -> DecisionResponse:
        """
        Executes the OpenJev decision contract across all question slots in parallel.
        Matches TypeSafe Jev & OpenJev schema.
        """
        t_start = time.perf_counter()
        state_str = str(req.state) if not isinstance(req.state, str) else req.state
        answers = {}
        all_detected_spans: List[DetectedSpan] = []

        for q in req.questions:
            q_type = q.type.lower()

            if q_type == "noul":
                # Noul: Binary Hypothesis Evaluation. Prompt is built by the SAME function used in training.
                prompt, raw_index = build_noul_prompt(state_str, q.question)
                want_spans = self.span_locator is not None
                encoding = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    return_offsets_mapping=want_spans
                )
                offsets = encoding.pop("offset_mapping")[0].tolist() if want_spans else None
                encoding = encoding.to(self.device)

                base_model = getattr(self.model, "model", self.model)
                outputs = base_model(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    output_hidden_states=True
                )
                target_ids = [self.false_token_id, self.true_token_id]
                zero = torch.zeros(1, dtype=torch.long, device=encoding.input_ids.device)
                last_pos = torch.full((1,), encoding.input_ids.size(1) - 1, dtype=torch.long,
                                      device=encoding.input_ids.device)
                candidate_logits = self.candidate_logits_fp32(outputs, zero, last_pos, target_ids)[0]
                candidate_logits = candidate_logits / max(self.temperature, 1e-3)
                probs = F.softmax(candidate_logits, dim=-1)

                entropy = self.calculate_entropy(probs)
                re_reads = 0

                # Active Re-Reads if uncertainty exceeds threshold
                if entropy > req.entropy_threshold and (req.samples or 1) > 1:
                    probs = self._execute_reread_perturbation(
                        prompt, target_ids, num_rereads=req.samples or 3
                    )
                    entropy = self.calculate_entropy(probs)
                    re_reads = req.samples or 3

                p_true = float(probs[1].item())
                confidence = max(0.0, min(1.0, 1.0 - entropy))

                answers[q.key] = NoulResult(
                    type="noul",
                    probability=round(p_true, 4),
                    confidence=round(confidence, 4),
                    value=(p_true >= self.doc_threshold),
                    re_reads_executed=re_reads
                )

                # Token span extraction (trained span head only; no regex substitutes)
                if want_spans:
                    span_logits = self.span_locator(outputs.hidden_states[-1].float())
                    span_probs = torch.sigmoid(span_logits).squeeze(0)
                    raw_ranges = token_raw_ranges([tuple(o) for o in offsets], raw_index)
                    all_detected_spans.extend(self.span_locator.extract_raw_spans(
                        token_probs=span_probs,
                        raw_ranges=raw_ranges,
                        raw_text=state_str,
                        threshold=self.span_threshold
                    ))

            elif q_type == "choice":
                # Choice: Constrained Categorical Routing
                assert q.options and len(q.options) > 0, "Choice primitive requires candidate options"
                opts = q.options

                option_token_ids = [resolve_single_token(self.tokenizer, opt) for opt in opts]
                opts_formatted = ", ".join(opts)
                prompt = (
                    f"State: {state_str}\n"
                    f"Question: {q.question}\n"
                    f"Options: [{opts_formatted}]\n"
                    f"Selection:"
                )
                encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                logits = self.extract_terminal_logits(encoding.input_ids, encoding.attention_mask)[0]
                candidate_logits = torch.stack([logits[tid] for tid in option_token_ids]) / max(self.temperature, 1e-3)
                probs = F.softmax(candidate_logits, dim=-1)

                entropy = self.calculate_entropy(probs)
                re_reads = 0

                if entropy > req.entropy_threshold and (req.samples or 1) > 1:
                    probs = self._execute_reread_perturbation(
                        prompt, option_token_ids, num_rereads=req.samples or 3
                    )
                    entropy = self.calculate_entropy(probs)
                    re_reads = req.samples or 3

                prob_dist = {opt: round(float(probs[i].item()), 4) for i, opt in enumerate(opts)}
                best_idx = int(torch.argmax(probs).item())
                confidence = max(0.0, min(1.0, 1.0 - entropy))

                answers[q.key] = ChoiceResult(
                    type="choice",
                    selection=opts[best_idx],
                    distribution=prob_dist,
                    confidence=round(confidence, 4),
                    re_reads_executed=re_reads
                )

            elif q_type == "score":
                # Score: Ordinal Threshold Expectation
                s_min = q.scale_min or 1
                s_max = q.scale_max or 5
                levels = [str(i) for i in range(s_min, s_max + 1)]
                level_token_ids = [resolve_single_token(self.tokenizer, lvl) for lvl in levels]

                prompt = (
                    f"State: {state_str}\n"
                    f"Question: {q.question}\n"
                    f"Rate on an ordinal scale from {s_min} to {s_max}:\n"
                    f"Score:"
                )
                encoding = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                logits = self.extract_terminal_logits(encoding.input_ids, encoding.attention_mask)[0]
                candidate_logits = torch.stack([logits[tid] for tid in level_token_ids])
                probs = F.softmax(candidate_logits, dim=-1)

                entropy = self.calculate_entropy(probs)
                re_reads = 0

                if entropy > req.entropy_threshold and (req.samples or 1) > 1:
                    probs = self._execute_reread_perturbation(
                        prompt, level_token_ids, num_rereads=req.samples or 3
                    )
                    re_reads = req.samples or 3

                # Compute mathematical expectation: E[score] = sum(m * P(level_m))
                expected_score = sum(int(lvl) * float(probs[i].item()) for i, lvl in enumerate(levels))
                prob_dist = {lvl: round(float(probs[i].item()), 4) for i, lvl in enumerate(levels)}

                answers[q.key] = ScoreResult(
                    type="score",
                    expected_value=round(expected_score, 3),
                    distribution=prob_dist,
                    confidence=round(max(0.0, min(1.0, 1.0 - entropy)), 4),
                    re_reads_executed=re_reads
                )

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return DecisionResponse(
            model=f"vaultgemma-openjev ({self.resolved_model_id})",
            answers=answers,
            latency_ms=round(t_elapsed_ms, 2),
            spans=all_detected_spans
        )

    def evaluate_text(
        self,
        text: str,
        question: str = "Contains HIPAA PHI or PII identifiers.",
        decision_threshold: Optional[float] = None
    ) -> NoulDecision:
        """
        High-level helper to evaluate a single clinical or general text string for PHI.
        Returns a NoulDecision containing calibrated probabilities, verdict, confidence, and spans.
        `decision_threshold` defaults to the engine's calibrated doc threshold (redactx_thresholds.json, else 0.5).
        """
        if decision_threshold is None:
            decision_threshold = self.doc_threshold
        t_start = time.perf_counter()
        req = DecisionRequest(
            state=text,
            questions=[
                QuestionPayload(
                    key="phi_eval",
                    type="noul",
                    question=question
                )
            ]
        )
        resp = self.evaluate_decision(req)
        ans = resp.answers.get("phi_eval")
        p_phi = ans.probability if ans else 0.0
        conf = ans.confidence if ans else 1.0
        is_phi = p_phi >= decision_threshold

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # Spans come only from the trained span head. If the checkpoint has no span head, spans is empty.
        spans = list(resp.spans or [])

        return NoulDecision(
            context=text,
            query=question,
            verdict="Contains_PHI_PII" if is_phi else "Clean",
            phi_probability=p_phi,
            clean_probability=round(1.0 - p_phi, 4),
            passed_gate=not is_phi,
            calibrated=True,
            confidence=conf,
            latency_ms=round(t_elapsed, 2),
            spans=spans,
            metadata={"confidence": conf}
        )

    @torch.inference_mode()
    def score_texts(
        self,
        texts: List[str],
        question: str = "Contains HIPAA PHI or PII identifiers.",
        batch_size: int = 8,
        span_threshold: Optional[float] = None,
        return_token_probs: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Batched Noul scoring. For each text returns:
          p_phi       calibrated P(PHI) (temperature applied)
          entropy     normalized Shannon entropy of the false/true distribution
          spans       List[DetectedSpan] in raw-text offsets (empty when the checkpoint has no span head)
          token_probs / raw_ranges  (only if return_token_probs) per-token span-head probabilities and
                      the raw-text range of each token ((-1, -1) for template tokens)
        Texts are right-padded; the verdict logit is read at each row's last real token, so results match
        unbatched evaluation up to floating-point noise.
        """
        thr = self.span_threshold if span_threshold is None else span_threshold
        want_spans = self.span_locator is not None
        results: List[Dict[str, Any]] = []
        prev_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "right"
        base_model = getattr(self.model, "model", self.model)
        try:
            step = max(1, batch_size)
            b = 0
            while b < len(texts):
                chunk = texts[b:b + step]
                built = [build_noul_prompt(t, question) for t in chunk]
                try:
                    enc = self.tokenizer(
                        [p for p, _ in built],
                        return_tensors="pt",
                        padding=True,
                        return_offsets_mapping=want_spans,
                    )
                    offsets = enc.pop("offset_mapping").tolist() if want_spans else None
                    enc = enc.to(self.device)
                    outputs = base_model(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        output_hidden_states=True,
                    )
                    last = enc.attention_mask.sum(dim=1) - 1
                    rows = torch.arange(enc.input_ids.size(0), device=enc.input_ids.device)
                    pair = self.candidate_logits_fp32(outputs, rows, last, [self.false_token_id, self.true_token_id])
                    probs = F.softmax(pair / max(self.temperature, 1e-3), dim=-1)
                    span_probs = (torch.sigmoid(self.span_locator(outputs.hidden_states[-1].float()))
                                  if want_spans else None)
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    if step > 1:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        step = max(1, step // 2)
                        continue
                    else:
                        raise

                for i, (text, (_, raw_index)) in enumerate(zip(chunk, built)):
                    rec: Dict[str, Any] = {
                        "p_phi": float(probs[i, 1].item()),
                        "entropy": self.calculate_entropy(probs[i]),
                        "spans": [],
                    }
                    if want_spans:
                        n_real = int(enc.attention_mask[i].sum().item())
                        row_offsets = [tuple(o) for o in offsets[i][:n_real]]
                        raw_ranges = token_raw_ranges(row_offsets, raw_index)
                        tp = span_probs[i, :n_real]
                        rec["spans"] = self.span_locator.extract_raw_spans(tp, raw_ranges, text, threshold=thr)
                        if return_token_probs:
                            rec["token_probs"] = tp.tolist()
                            rec["raw_ranges"] = raw_ranges
                    results.append(rec)
                b += step
        finally:
            self.tokenizer.padding_side = prev_side
        return results
    def save_pretrained(self, output_dir: str = "./models/redactx"):
        """Saves LoRA adapters, span locator, metadata, and scores into the models directory."""
        import json
        os.makedirs(output_dir, exist_ok=True)

        meta = {
            "engine": "OpenJevVaultGemmaEngine",
            "base_model_id": self.config.base_model_id,
            "resolved_model_id": self.resolved_model_id,
            "hidden_dim": self.hidden_dim,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "true_token_id": self.true_token_id,
            "false_token_id": self.false_token_id,
            "enable_span_locator": self.span_locator is not None
        }

        with open(os.path.join(output_dir, "redactx_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.scores:
            with open(os.path.join(output_dir, "scores.json"), "w", encoding="utf-8") as f:
                json.dump(self.scores, f, indent=2)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(os.path.join(output_dir, "lora_adapter"))

        if self.span_locator is not None:
            torch.save(self.span_locator.state_dict(), os.path.join(output_dir, "redactx_span_locator.pt"))

        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"OpenJev VaultGemma model saved to {output_dir}")

    def load_pretrained(self, model_path: str = "./models/redactx"):
        """Loads trained LoRA adapters, span locator, and scores from model directory."""
        import json
        lora_dir = os.path.join(model_path, "lora_adapter")
        if os.path.exists(lora_dir) and hasattr(self.model, "load_adapter"):
            try:
                self.model.load_adapter(lora_dir, "default")
                logger.info("Loaded LoRA adapter weights.")
            except Exception as e:
                logger.warning(f"Could not load LoRA adapter: {e}")

        span_path = os.path.join(model_path, "redactx_span_locator.pt")
        if os.path.exists(span_path) and self.span_locator is not None:
            self.span_locator.load_state_dict(torch.load(span_path, map_location=self.device))
            logger.info("Loaded span locator weights.")

        scores_path = os.path.join(model_path, "scores.json")
        if os.path.exists(scores_path):
            with open(scores_path, "r", encoding="utf-8") as f:
                self.scores = json.load(f)
            logger.info("Loaded scores.")

    @torch.inference_mode()
    def evaluate_redaction(self, state: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the exact RedactX decision contract specified in Section 4:
        Evaluates single-pass binary logits for 'true' vs 'false',
        computes Shannon entropy, and returns action ('REDACT', 'ESCALATE', 'PASS').
        """
        import json
        state_str = json.dumps(state) if isinstance(state, dict) else str(state)
        prompt = (
            f"[STATE]: {state_str}\n"
            f"[QUERY]: Does this text contain Protected Health Information or Personal Identifiable Information?\n"
            f"[EVALUATION]:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        if hasattr(outputs, "logits"):
            logits = outputs.logits[0, -1, :]
        else:
            logits = self.readout_head(outputs.last_hidden_state[0, -1, :])

        binary_logits = torch.stack([logits[self.false_token_id], logits[self.true_token_id]])
        probs = F.softmax(binary_logits, dim=-1)
        p_phi = float(probs[1].item())

        # Calculate Shannon entropy
        entropy = -float(torch.sum(probs * torch.log(probs + 1e-9)).item())

        return {
            "is_phi_pii": p_phi >= 0.70,
            "phi_probability": round(p_phi, 4),
            "calibration_entropy": round(entropy, 4),
            "action": "REDACT" if p_phi >= 0.70 else ("ESCALATE" if entropy > 0.45 else "PASS")
        }

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "./models/RedactX",
        device: Optional[str] = None
    ) -> "OpenJevVaultGemmaEngine":
        """Single-line factory loader to use trained OpenJev model in code."""
        import json
        if not os.path.exists(model_path):
            for candidate in ["./models/RedactX", "./models/redactx", "./models/openjev_vaultgemma"]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        meta_file = os.path.join(model_path, "redactx_meta.json")
        config_file = os.path.join(model_path, "config.json")

        if os.path.exists(config_file):
            # Model was exported as a standalone merged causal LM
            config = RedactXConfig(
                base_model_id=model_path,
                device=device,
                use_fallback_if_gated=False
            )
            instance = cls(config=config)
            scores_file = os.path.join(model_path, "scores.json")
            if os.path.exists(scores_file):
                with open(scores_file, "r", encoding="utf-8") as f:
                    instance.scores = json.load(f)
            instance.model_name = getattr(instance, "scores", {}).get("model_name", "RedactX")
            return instance

        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            config = RedactXConfig(
                base_model_id=meta.get("resolved_model_id", meta.get("base_model_id")),
                hidden_dim=meta.get("hidden_dim"),
                lora_r=meta.get("lora_r", 16),
                lora_alpha=meta.get("lora_alpha", 32),
                enable_span_locator=meta.get("enable_span_locator", True),
                device=device
            )
        else:
            config = RedactXConfig(device=device)

        instance = cls(config=config)
        instance.load_pretrained(model_path)
        return instance
