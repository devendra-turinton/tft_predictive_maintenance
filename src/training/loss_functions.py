import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class EarlyFailureLoss(torch.nn.Module):
    """
    Custom loss function that penalizes more for errors when predicting failures closer to the present.
    This is critical for predictive maintenance where early detection is more important.
    """
    def __init__(self, alpha: float = 2.0, beta: float = 0.5, threshold: float = 0.5):
        """
        Initialize the loss function.
        
        Args:
            alpha: Weight factor for predictions close to failure
            beta: Base weight for all predictions
            threshold: Threshold below which failures are considered imminent
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighted loss.
        
        Args:
            y_pred: Predicted values [batch_size, sequence_length]
            y_true: True values [batch_size, sequence_length]
            
        Returns:
            Weighted loss value
        """
        # Calculate base loss (MAE)
        base_loss = torch.abs(y_pred - y_true)
        
        # Calculate weights based on proximity to failure
        # Higher weight for predictions where true value is close to failure
        weights = self.beta + self.alpha * torch.exp(-y_true / self.threshold)
        
        # Apply weights and calculate mean
        weighted_loss = (base_loss * weights).mean()
        
        return weighted_loss

class QuantileLossWithFailurePenalty(torch.nn.Module):
    """
    Combines quantile loss with a penalty for missing imminent failures.
    Important for predictive maintenance to reduce false negatives.
    """
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.5, 0.9], 
        failure_threshold: float = 0.2,
        failure_penalty: float = 5.0
    ):
        """
        Initialize the loss function.
        
        Args:
            quantiles: List of quantiles to predict
            failure_threshold: Threshold below which failures are considered imminent
            failure_penalty: Penalty multiplier for missing imminent failures
        """
        super().__init__()
        self.quantiles = quantiles
        self.failure_threshold = failure_threshold
        self.failure_penalty = failure_penalty
        
    def forward(self, y_pred: Dict[float, torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the combined quantile loss with failure penalty.
        
        Args:
            y_pred: Dictionary of predictions for each quantile
            y_true: True values
            
        Returns:
            Combined loss value
        """
        # Calculate quantile loss for each quantile
        quantile_losses = []
        
        for q, pred in y_pred.items():
            # Calculate the quantile loss
            errors = y_true - pred
            quantile_loss = torch.max((q - 1) * errors, q * errors)
            quantile_losses.append(quantile_loss.mean())
        
        # Calculate base loss as mean of quantile losses
        base_loss = torch.stack(quantile_losses).mean()
        
        # Add penalty for missing imminent failures
        # Check if any true values are below failure threshold
        imminent_failures = y_true < self.failure_threshold
        
        if torch.any(imminent_failures):
            # Get median predictions for imminent failures
            median_pred = y_pred.get(0.5, next(iter(y_pred.values())))
            median_pred_for_failures = median_pred[imminent_failures]
            
            # True values for imminent failures
            true_for_failures = y_true[imminent_failures]
            
            # Calculate penalty for missing imminent failures
            # Penalize more for predicting higher values when failure is imminent
            failure_errors = torch.relu(median_pred_for_failures - true_for_failures)
            failure_penalty = self.failure_penalty * failure_errors.mean()
            
            # Add penalty to base loss
            total_loss = base_loss + failure_penalty
        else:
            total_loss = base_loss
        
        return total_loss