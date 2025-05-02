import torch
import numpy as np
import os
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pytorch_lightning as pl
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
import pandas as pd
import joblib
from ..data.dataset import PredictiveMaintenanceDataset

logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path: str, config: Dict) -> TemporalFusionTransformer:
    """
    Load a TFT model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Model configuration
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the model
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    
    logger.info("Model loaded successfully")
    return model

def save_model_config(config: Dict, output_dir: str, filename: str = "model_config.yaml") -> None:
    """
    Save model configuration to a file.
    
    Args:
        config: Model configuration
        output_dir: Directory to save to
        filename: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved model configuration to {output_path}")

def get_best_model_path(checkpoint_dir: str) -> str:
    """
    Get the path to the best model checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the best checkpoint
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Find the checkpoint with the best validation loss
    # Assume format is like: 'epoch=X-val_loss=Y.ckpt'
    best_val_loss = float('inf')
    best_checkpoint = None
    
    for ckpt in checkpoints:
        try:
            # Extract validation loss from filename
            val_loss = float(ckpt.split('val_loss=')[1].split('.')[0])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt
        except (IndexError, ValueError):
            # If filename doesn't follow expected format, skip it
            continue
    
    if best_checkpoint is None:
        # If we couldn't parse validation losses, just take the latest checkpoint
        best_checkpoint = sorted(checkpoints)[-1]
    
    logger.info(f"Selected best checkpoint: {best_checkpoint}")
    return os.path.join(checkpoint_dir, best_checkpoint)

def setup_trainer(config: Dict) -> pl.Trainer:
    """
    Set up PyTorch Lightning trainer with configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        PyTorch Lightning Trainer
    """
    # Set up callbacks
    callbacks = []
    
    # Early stopping callback
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.get("early_stopping_patience", 5),
        mode="min"
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.get("checkpoint_dir", "checkpoints"),
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min"
    )
    callbacks.append(checkpoint_callback)
    
    # LR Monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Set up loggers
    loggers = []
    
    # TensorBoard logger
    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=config.get("log_dir", "logs"),
        name="tft_predictive_maintenance"
    )
    loggers.append(tensorboard_logger)
    
    # Weights & Biases logger if configured
    if config.get("use_wandb", False):
        wandb_logger = pl.loggers.WandbLogger(
            project=config.get("wandb_project", "predictive_maintenance_tft"),
            log_model=True
        )
        loggers.append(wandb_logger)
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 50),
        accelerator="gpu" if config.get("gpus", 0) > 0 else "cpu",
        devices=config.get("gpus", 0) if config.get("gpus", 0) > 0 else None,
        gradient_clip_val=config.get("gradient_clip_val", 0.1),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.get("log_interval", 10),
        precision=config.get("precision", 16) if config.get("gpus", 0) > 0 else 32,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        deterministic=True
    )
    
    logger.info("Set up PyTorch Lightning trainer")
    return trainer

def create_prediction_dataset(
    df: pd.DataFrame, 
    training_dataset: TimeSeriesDataSet,
    predict_steps: int
) -> TimeSeriesDataSet:
    """
    Create a dataset for making predictions.
    
    Args:
        df: DataFrame with data for prediction
        training_dataset: Original training dataset to base parameters on
        predict_steps: Number of steps to predict
        
    Returns:
        TimeSeriesDataSet for prediction
    """
    # Create a prediction dataset based on the training dataset
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, 
        df, 
        predict=True,
        stop_randomization=True,
        max_prediction_length=predict_steps
    )
    
    logger.info(f"Created prediction dataset with {len(prediction_dataset)} samples")
    return prediction_dataset

def interpret_predictions(
    predictions: Dict[float, torch.Tensor], 
    target_scaler: Any, 
    prediction_times: pd.DatetimeIndex,
    quantiles: List[float] = [0.1, 0.5, 0.9]
) -> pd.DataFrame:
    """
    Interpret model predictions and convert to meaningful values.
    
    Args:
        predictions: Dictionary of predictions for different quantiles
        target_scaler: Scaler used for the target variable
        prediction_times: DatetimeIndex of prediction times
        quantiles: List of quantiles used in predictions
        
    Returns:
        DataFrame with interpreted predictions
    """
    # Convert predictions to numpy arrays and inverse transform
    results = {}
    
    for q in quantiles:
        if q in predictions:
            # Get prediction for this quantile
            pred = predictions[q].detach().cpu().numpy()
            
            # Reshape if needed
            if len(pred.shape) == 3:  # [batch, time, 1]
                pred = pred.squeeze(-1)
                
            # Inverse transform if needed
            if target_scaler is not None:
                # We need to reshape for scikit-learn scalers
                original_shape = pred.shape
                pred = pred.reshape(-1, 1)
                pred = target_scaler.inverse_transform(pred)
                pred = pred.reshape(original_shape)
            
            # Store in results
            results[f"q{int(q*100)}"] = pred
    
    # Create a DataFrame with predictions
    result_df = pd.DataFrame()
    
    # Add time index
    if len(prediction_times) == len(results[list(results.keys())[0]]):
        result_df["timestamp"] = prediction_times
    else:
        # If lengths don't match, create a sequence of timestamps
        result_df["timestamp"] = pd.date_range(
            start=prediction_times[0],
            periods=len(results[list(results.keys())[0]]),
            freq="infer"
        )
    
    # Add each quantile prediction
    for q_name, values in results.items():
        if len(values.shape) == 1:
            # Single step prediction
            result_df[q_name] = values
        else:
            # Multi-step prediction - flatten into separate columns
            for i in range(values.shape[1]):
                result_df[f"{q_name}_step{i+1}"] = values[:, i]
    
    logger.info("Processed predictions into DataFrame")
    return result_df