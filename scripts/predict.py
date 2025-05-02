#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import torch
import json
from datetime import datetime

# Add the project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.common_utils import (
    setup_logging, load_config, save_predictions, check_gpu_availability
)
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import PredictiveMaintenanceDataset
from src.models.model_utils import (
    load_model_from_checkpoint, get_best_model_path, 
    create_prediction_dataset, interpret_predictions
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate predictions with trained TFT model")
    
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory with trained model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Specific checkpoint to use (if None, uses best)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data for prediction")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save predictions (defaults to exp_dir/predictions.csv)")
    parser.add_argument("--output_format", type=str, default="csv",
                        choices=["csv", "parquet", "json"],
                        help="Format for saving predictions")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Prediction horizon (if different from training config)")
    parser.add_argument("--interpret", action="store_true",
                        help="Generate and save interpretation data")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage even if available")
    
    return parser.parse_args()

def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logs_dir = os.path.join(args.exp_dir, "logs")
    setup_logging(logs_dir, level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting prediction for experiment: {args.exp_dir}")
    
    # Set output path
    if not args.output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = os.path.join(args.exp_dir, f"predictions_{timestamp}.{args.output_format}")
    
    # Load configuration
    config_path = os.path.join(args.exp_dir, "config.yaml")
    config = load_config(config_path)
    
    # Update prediction horizon if specified
    if args.horizon:
        config["data"]["max_prediction_length"] = args.horizon
        logger.info(f"Updated prediction horizon to {args.horizon}")
    
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
    logger.info(f"Loading data from {args.data_path}")
    config["data"]["csv_file_path"] = args.data_path
    data_loader = DataLoader(config["data"])
    df = data_loader.load_data()
    
    # Load preprocessor
    preprocessor_dir = os.path.join(args.exp_dir, "preprocessor")
    preprocessor = DataPreprocessor.load_preprocessor(preprocessor_dir, config["data"])
    
    # Preprocess data
    df = preprocessor.fit_transform(df, is_training=False)
    
    # Create training dataset first (needed for creating prediction dataset)
    dataset_creator = PredictiveMaintenanceDataset(config["data"])
    
    # We need to create a dummy training dataset to get the parameters
    # Split the data into three equal parts
    n = len(df)
    train_size = n // 3
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:2*train_size]
    test_df = df