import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

logger = logging.getLogger(__name__)

class PredictiveMaintenanceMetrics:
    """
    Class for calculating and tracking metrics specific to predictive maintenance.
    """
    def __init__(self, config: Dict):
        """
        Initialize metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metric_names = config.get("metrics", ["mae", "mse", "mape", "smape"])
        
    def calculate_metrics(
        self, 
        predictions: Union[np.ndarray, pd.DataFrame],
        targets: Union[np.ndarray, pd.DataFrame],
        prediction_times: Optional[pd.DatetimeIndex] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate metrics for predictive maintenance.
        
        Args:
            predictions: Model predictions (median predictions if probabilistic)
            targets: True target values
            prediction_times: Optional timestamps for each prediction
            threshold: Threshold for binary classification metrics
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays if needed
        if isinstance(predictions, pd.DataFrame):
            # If predictions is a DataFrame with multiple columns (quantiles),
            # use the median prediction (q50)
            if "q50" in predictions.columns:
                predictions = predictions["q50"].values
            else:
                predictions = predictions.values
                
        if isinstance(targets, pd.DataFrame):
            targets = targets.values
        
        # Ensure shapes match
        if predictions.shape != targets.shape:
            raise ValueError(f"Prediction shape {predictions.shape} doesn't match target shape {targets.shape}")
        
        metrics = {}
        
        # Calculate standard regression metrics
        if "mae" in self.metric_names:
            metrics["mae"] = mean_absolute_error(targets, predictions)
            
        if "mse" in self.metric_names:
            metrics["mse"] = mean_squared_error(targets, predictions)
            
        if "rmse" in self.metric_names:
            metrics["rmse"] = math.sqrt(mean_squared_error(targets, predictions))
            
        if "r2" in self.metric_names:
            metrics["r2"] = r2_score(targets, predictions)
            
        if "mape" in self.metric_names:
            # Avoid division by zero
            mask = targets != 0
            if np.any(mask):
                metrics["mape"] = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
            else:
                metrics["mape"] = np.nan
                
        if "smape" in self.metric_names:
            # Symmetric Mean Absolute Percentage Error
            denominator = np.abs(targets) + np.abs(predictions)
            mask = denominator != 0
            if np.any(mask):
                metrics["smape"] = np.mean(2.0 * np.abs(predictions[mask] - targets[mask]) / denominator[mask]) * 100
            else:
                metrics["smape"] = np.nan
        
        # Calculate predictive maintenance specific metrics
        
        # Early Detection Rate - how often we predict failure before it happens
        if "early_detection_rate" in self.metric_names and prediction_times is not None:
            # This metric requires time information
            # For each actual failure, check if we predicted it early enough
            metrics["early_detection_rate"] = self._calculate_early_detection_rate(
                predictions, targets, prediction_times, threshold
            )
        
        # False Alarm Rate - how often we predict a failure that doesn't happen
        if "false_alarm_rate" in self.metric_names:
            metrics["false_alarm_rate"] = self._calculate_false_alarm_rate(
                predictions, targets, threshold
            )
        
        # Mean Time Between Failures (MTBF) Error
        if "mtbf_error" in self.metric_names and prediction_times is not None:
            metrics["mtbf_error"] = self._calculate_mtbf_error(
                predictions, targets, prediction_times, threshold
            )
            
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
    
    def _calculate_early_detection_rate(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        prediction_times: pd.DatetimeIndex,
        threshold: float
    ) -> float:
        """
        Calculate the early detection rate.
        
        Args:
            predictions: Model predictions
            targets: True target values
            prediction_times: Timestamps for each prediction
            threshold: Threshold for failure detection
            
        Returns:
            Early detection rate (0-1)
        """
        # Find actual failures in targets
        actual_failures = targets <= threshold
        
        if not np.any(actual_failures):
            # No actual failures to detect
            return 1.0
        
        # Find predicted failures
        predicted_failures = predictions <= threshold
        
        # Count correctly predicted failures
        true_positives = np.logical_and(actual_failures, predicted_failures)
        
        # Calculate early detection rate
        early_detection_rate = np.sum(true_positives) / np.sum(actual_failures)
        
        return early_detection_rate
    
    def _calculate_false_alarm_rate(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float
    ) -> float:
        """
        Calculate the false alarm rate.
        
        Args:
            predictions: Model predictions
            targets: True target values
            threshold: Threshold for failure detection
            
        Returns:
            False alarm rate (0-1)
        """
        # Find predicted failures
        predicted_failures = predictions <= threshold
        
        if not np.any(predicted_failures):
            # No predicted failures
            return 0.0
        
        # Find actual non-failures
        actual_non_failures = targets > threshold
        
        # Count false alarms
        false_alarms = np.logical_and(predicted_failures, actual_non_failures)
        
        # Calculate false alarm rate
        false_alarm_rate = np.sum(false_alarms) / np.sum(predicted_failures)
        
        return false_alarm_rate
    
    def _calculate_mtbf_error(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        prediction_times: pd.DatetimeIndex,
        threshold: float
    ) -> float:
        """
        Calculate the Mean Time Between Failures (MTBF) error.
        
        Args:
            predictions: Model predictions
            targets: True target values
            prediction_times: Timestamps for each prediction
            threshold: Threshold for failure detection
            
        Returns:
            MTBF error (percentage)
        """
        # Convert prediction times to timedeltas
        if len(prediction_times) > 1:
            time_diffs = np.diff(prediction_times.values).astype('timedelta64[h]').astype(float)
            avg_time_step = np.mean(time_diffs)
        else:
            # Default to 1 hour if we can't calculate
            avg_time_step = 1.0
        
        # Find actual failures in targets
        actual_failures = np.where(targets <= threshold)[0]
        
        if len(actual_failures) <= 1:
            # Not enough failures to calculate MTBF
            return 0.0
        
        # Calculate actual MTBF
        actual_mtbf = np.mean(np.diff(actual_failures)) * avg_time_step
        
        # Find predicted failures
        predicted_failures = np.where(predictions <= threshold)[0]
        
        if len(predicted_failures) <= 1:
            # Not enough predicted failures to calculate MTBF
            return 100.0
        
        # Calculate predicted MTBF
        predicted_mtbf = np.mean(np.diff(predicted_failures)) * avg_time_step
        
        # Calculate MTBF error
        if actual_mtbf == 0:
            return 100.0
        
        mtbf_error = abs(predicted_mtbf - actual_mtbf) / actual_mtbf * 100
        
        return mtbf_error
        
    def calculate_lead_time(
        self, 
        predictions: np.ndarray,
        targets: np.ndarray,
        prediction_times: pd.DatetimeIndex,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate the average lead time for failure predictions.
        
        Args:
            predictions: Model predictions
            targets: True target values
            prediction_times: Timestamps for each prediction
            threshold: Threshold for failure detection
            
        Returns:
            Dictionary with lead time metrics
        """
        # Convert prediction times to numpy datetime64
        times = np.array(prediction_times)
        
        # Find actual failures
        actual_failure_indices = np.where(targets <= threshold)[0]
        
        if len(actual_failure_indices) == 0:
            return {"avg_lead_time": 0, "min_lead_time": 0, "max_lead_time": 0}
        
        # Calculate lead times for each failure
        lead_times = []
        
        for failure_idx in actual_failure_indices:
            # Find the earliest prediction of this failure
            # Look for predictions below threshold before the actual failure
            pred_failure_indices = np.where(
                (predictions <= threshold) & 
                (np.arange(len(predictions)) < failure_idx)
            )[0]
            
            if len(pred_failure_indices) > 0:
                # Get the earliest prediction
                earliest_pred_idx = pred_failure_indices[0]
                
                # Calculate lead time in hours
                lead_time = (times[failure_idx] - times[earliest_pred_idx]).astype('timedelta64[h]').astype(float)
                lead_times.append(lead_time)
        
        if len(lead_times) == 0:
            return {"avg_lead_time": 0, "min_lead_time": 0, "max_lead_time": 0}
        
        # Calculate lead time metrics
        lead_time_metrics = {
            "avg_lead_time": np.mean(lead_times),
            "min_lead_time": np.min(lead_times),
            "max_lead_time": np.max(lead_times)
        }
        
        return lead_time_metrics