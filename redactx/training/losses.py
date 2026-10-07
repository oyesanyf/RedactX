"""
OpenJev Composite Calibration Loss (RLCD).
Combines Soft-Target KL Divergence, Brier Score (MSE), and Permutation Consistency Loss.
Penalizes miscalibration, extreme overconfidence, and option order bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JevCalibrationLoss(nn.Module):
    """
    Composite Calibration Loss for OpenJev:
    - Soft-target Cross-Entropy via KL Divergence on valid candidate token logits
    - Brier Score (Mean Squared Error on predicted vs ground-truth probabilities)
    - Consistency Loss across shuffled option permutations
    """

    def __init__(
        self,
        lambda_brier: float = 1.0,
        lambda_consistency: float = 0.5,
        lambda_ece: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.lambda_brier = lambda_brier
        self.lambda_consistency = lambda_consistency
        self.lambda_ece = lambda_ece

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_dist: torch.Tensor,
        alt_pred_logits: torch.Tensor = None,
        candidate_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: Logits gathered over valid candidate token IDs [batch_size, num_options]
            target_dist: Soft ground-truth distribution [batch_size, num_options]
            alt_pred_logits: Optional logits from permuted/shuffled options [batch_size, num_options]
            candidate_mask: Optional boolean mask [batch_size, num_options] for valid options
        Returns:
            total_loss = loss_ce + lambda_brier * loss_brier + lambda_consistency * loss_consistency
        """
        if candidate_mask is not None:
            pred_logits = pred_logits.masked_fill(~candidate_mask, -1e9)
            if alt_pred_logits is not None:
                alt_pred_logits = alt_pred_logits.masked_fill(~candidate_mask, -1e9)

        # Ensure target_dist is a float distribution; if 1D class indices are provided, convert to one-hot
        if target_dist.dim() == 1:
            target_dist = F.one_hot(target_dist.long(), num_classes=pred_logits.size(-1)).float()
        elif not torch.is_floating_point(target_dist):
            target_dist = target_dist.float()

        pred_probs = F.softmax(pred_logits, dim=-1)
        log_probs = F.log_softmax(pred_logits, dim=-1)

        # Soft-target Cross-Entropy / KL Divergence
        # Note: KLDiv expects log-probabilities for input and probabilities for target
        loss_ce = F.kl_div(log_probs, target_dist, reduction="batchmean")

        # Brier Score (Mean Squared Error on probabilities)
        loss_brier = torch.mean(torch.sum((pred_probs - target_dist) ** 2, dim=-1))

        total_loss = loss_ce + (self.lambda_brier * loss_brier)

        # Consistency loss across shuffled permutations
        if alt_pred_logits is not None:
            alt_probs = F.softmax(alt_pred_logits, dim=-1)
            loss_consistency = F.mse_loss(pred_probs, alt_probs)
            total_loss = total_loss + (self.lambda_consistency * loss_consistency)

        return total_loss


# Alias for backward compatibility
RLCDCalibrationLoss = JevCalibrationLoss
