#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import torch

# Add the project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.common_utils import (
    setup_logging, load_config, save_metrics, check_gpu_availability
)
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import PredictiveMaintenanceDataset
from src.models.model_utils import load_model_from_checkpoint, get_best_model_path
from src.evaluation.metrics import PredictiveMaintenanceMetrics
from src.evaluation.visualization import PredictiveMaintenanceVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TFT model for predictive maintenance")
    
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory with trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint to use (if None, uses best)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to test data (if different from training)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to exp_dir/results)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage even if available")
    
    return parser.parse_args()

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(os.path.join(args.exp_dir, "logs"), level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation for experiment: {args.exp_dir}")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.exp_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config_path = os.path.join(args.exp_dir, "config.yaml")
    config = load_config(config_path)
    
    # Check GPU availability
    if not args.no_gpu:
        gpu_available = check_gpu_availability()
        if gpu_available:
            config["training"]["gpus"] = 1
        else:
            config["training"]["gpus"] = 0
    else:
        logger.info("GPU usage disabled by command line argument")
        config["training"]["gpus"] = 0
    
    # Load data
    if args.data_path:
        config["data"]["csv_file_path"] = args.data_path
    
    logger.info(f"Loading data from {config['data']['csv_file_path']}")
    data_loader = DataLoader(config["data"])
    _, _, test_df = data_loader.get_data_splits()
    
    # Load preprocessor
    preprocessor_dir = os.path.join(args.exp_dir, "preprocessor")
    preprocessor = DataPreprocessor.load_preprocessor(preprocessor_dir, config["data"])
    
    # Preprocess test data
    test_df = preprocessor.fit_transform(test_df, is_training=False)
    
    # Create dataset
    dataset_creator = PredictiveMaintenanceDataset(config["data"])
    
    # First create a temporary training dataset to get the parameters
    temp_train_df = test_df.copy()  # Use test data as dummy training data
    train_dataset, _, _ = dataset_creator.create_datasets(temp_train_df, temp_train_df, temp_train_df)
    
    # Now create a proper test dataset
    test_dataset = dataset_creator.create_datasets(test_df, test_df, test_df)[2]
    test_dataloader = dataset_creator.create_data_loaders(None, None, test_dataset)[2]
    
    # Get dataset parameters
    dataset_params = dataset_creator.get_parameters()
    
    # Load model
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = os.path.join(args.exp_dir, "checkpoints")
        checkpoint_path = get_best_model_path(checkpoint_dir)
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, config["model"])
    
    # Move model to correct device
    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()
    
    # Set up metrics calculator
    metrics_calculator = PredictiveMaintenanceMetrics(config["evaluation"])
    
    # Set up visualizer if needed
    if args.visualize:
        visualizer = PredictiveMaintenanceVisualizer(config["evaluation"])
        visualization_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(visualization_dir, exist_ok=True)
    
    # Perform evaluation
    logger.info("Starting evaluation")
    
    # Get predictions
    all_predictions = []
    all_targets = []
    all_timestamps = []
    all_machine_ids = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get predictions
            predictions = model.predict(
                batch,
                return_quantiles=config["evaluation"].get("confidence_intervals", [50, 90])
            )
            
            # Extract targets
            targets = batch["target"]
            
            # Extract timestamps if available
            if "timestamp" in batch:
                timestamps = batch["timestamp"]
            else:
                timestamps = torch.arange(len(targets))
            
            # Extract machine IDs if available
            if "machine_id" in batch:
                machine_ids = batch["machine_id"]
            else:
                machine_ids = torch.zeros(len(targets))
            
            # Add to lists
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_timestamps.append(timestamps)
            all_machine_ids.append(machine_ids)
    
    # Concatenate all predictions and targets
    all_predictions = {q: torch.cat([pred[q] for pred in all_predictions], dim=0) for q in all_predictions[0].keys()}
    all_targets = torch.cat(all_targets, dim=0)
    all_timestamps = torch.cat(all_timestamps, dim=0)
    all_machine_ids = torch.cat(all_machine_ids, dim=0)
    
    # Convert to numpy arrays
    all_predictions = {q: pred.cpu().numpy() for q, pred in all_predictions.items()}
    all_targets = all_targets.cpu().numpy()
    all_timestamps = all_timestamps.cpu().numpy()
    all_machine_ids = all_machine_ids.cpu().numpy()
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics")
    metrics = metrics_calculator.calculate_metrics(
        all_predictions[0.5] if 0.5 in all_predictions else next(iter(all_predictions.values())),
        all_targets,
        pd.DatetimeIndex(all_timestamps) if "timestamp" in batch else None
    )
    
    # Calculate lead time metrics if timestamps are available
    if "timestamp" in batch:
        lead_time_metrics = metrics_calculator.calculate_lead_time(
            all_predictions[0.5] if 0.5 in all_predictions else next(iter(all_predictions.values())),
            all_targets,
            pd.DatetimeIndex(all_timestamps)
        )
        metrics.update(lead_time_metrics)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    # Create visualizations if requested
    if args.visualize:
        logger.info("Creating visualizations")
        
        # Plot predictions vs. actual
        visualizer.plot_predictions(
            all_predictions,
            all_targets,
            pd.DatetimeIndex(all_timestamps) if "timestamp" in batch else pd.RangeIndex(len(all_targets)),
            all_machine_ids if "machine_id" in batch else None,
            output_dir=visualization_dir
        )
        
        # Get feature importances if available
        try:
            # Select a small batch for interpretation
            sample_batch = next(iter(test_dataloader))
            sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
            
            # Get interpretation data
            interpretation = model.get_interpretation(sample_batch)
            
            # Plot feature importance
            visualizer.plot_feature_importance(
                interpretation,
                output_dir=visualization_dir
            )
            
            # Plot attention patterns if available
            try:
                attention_weights = model.get_attention_weights(sample_batch)
                visualizer.plot_attention_patterns(
                    attention_weights,
                    pd.DatetimeIndex(sample_batch["timestamp"]) if "timestamp" in sample_batch else None,
                    output_dir=visualization_dir
                )
            except Exception as e:
                logger.warning(f"Could not create attention pattern plots: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Could not create interpretation plots: {str(e)}")
    
    logger.info("Evaluation completed")
    logger.info(f"Results saved to {output_dir}")
    
    # Print summary of metrics
    logger.info("Evaluation metrics summary:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()