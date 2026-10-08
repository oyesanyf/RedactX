"""
Hobson Pointer Scorer Head and Platt Temperature Calibrator for RedactX.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RedactXPointerScorer(nn.Module):
    """
    Single-pass scoring pointer head based on the Strands Decider Hobson architecture.
    Projects pooled token representations through non-linear transformation, normalization,
    and dropout before classification.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.projection = nn.Linear(hidden_dim, hidden_dim // 2)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_embedding: [batch_size, hidden_dim]
        Returns:
            logits: [batch_size, num_classes]
        """
        projected = self.projection(pooled_embedding)
        activated = self.activation(projected)
        normalized = self.norm(activated)
        dropped = self.dropout(normalized)
        return self.classifier(dropped)


class PlattTemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibrator (Platt scaling for modern deep neural networks).

    Mathematical Formulation:
        Given uncalibrated logits z_i \\in \\mathbb{R}^K for sample i, Platt temperature
        scaling applies a single scalar parameter T > 0:

        \\hat{p}_{i,k} = \\frac{\\exp(z_{i,k} / T)}{\\sum_{j=1}^K \\exp(z_{i,j} / T)}

        The parameter T is optimized on a held-out validation set to minimize negative log-likelihood (NLL):

        T^* = \\arg\\min_{T > 0} \\; -\\frac{1}{N} \\sum_{i=1}^N \\sum_{k=1}^K y_{i,k} \\log \\hat{p}_{i,k}(T)

        Because T does not change the argmax ordering of logits (monotonic transformation),
        temperature scaling preserves accuracy and AUROC while minimizing Expected Calibration Error (ECE).
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by current temperature."""
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities."""
        scaled = self.forward(logits)
        return F.softmax(scaled, dim=-1)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimizes temperature on a validation split using L-BFGS to minimize Cross-Entropy.
        """
        val_logits = val_logits.detach()
        val_labels = val_labels.detach()
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())
