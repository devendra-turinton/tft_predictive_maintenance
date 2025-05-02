#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import yaml

# Add the project directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.common_utils import (
    setup_logging, set_seed, load_config, save_config,
    create_experiment_dir, update_config_paths, check_gpu_availability
)
from src.data.data_loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.data.dataset import PredictiveMaintenanceDataset
from src.training.trainer import ModelTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TFT model for predictive maintenance")
    
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--data_path", type=str, 
                        help="Path to data file (overrides config)")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Experiment directory (if None, creates timestamped directory)")
    parser.add_argument("--hp_opt", action="store_true", 
                        help="Perform hyperparameter optimization")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of hyperparameter optimization trials")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Disable GPU usage even if available")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create experiment directory
    exp_dir = args.exp_dir if args.exp_dir else create_experiment_dir()
    
    # Setup logging
    setup_logging(os.path.join(exp_dir, "logs"), level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with experiment directory: {exp_dir}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    if args.data_path:
        config["data"]["csv_file_path"] = args.data_path
    
    if args.seed:
        config["training"]["seed"] = args.seed
    
    # Update configuration paths for experiment
    config = update_config_paths(config, exp_dir)
    
    # Save updated configuration
    save_config(config, os.path.join(exp_dir, "config.yaml"))
    
    # Set random seed
    set_seed(config["training"]["seed"])
    
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
    
    # Load and prepare data
    logger.info("Loading and preparing data")
    data_loader = DataLoader(config["data"])
    train_df, val_df, test_df = data_loader.get_data_splits()
    
    # Preprocess data
    preprocessor = DataPreprocessor(config["data"])
    train_df = preprocessor.fit_transform(train_df, is_training=True)
    val_df = preprocessor.fit_transform(val_df, is_training=False)
    test_df = preprocessor.fit_transform(test_df, is_training=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor(os.path.join(exp_dir, "preprocessor"))
    
    # Create datasets
    dataset_creator = PredictiveMaintenanceDataset(config["data"])
    train_dataset, val_dataset, test_dataset = dataset_creator.create_datasets(
        train_df, val_df, test_df
    )
    
    # Create data loaders
    train_dataloader, val_dataloader, test_dataloader = dataset_creator.create_data_loaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Get dataset parameters for model
    dataset_params = dataset_creator.get_parameters()
    
    # Initialize trainer
    model_trainer = ModelTrainer(config, dataset_params)
    
    # Training process
    if args.hp_opt:
        # Perform hyperparameter optimization
        logger.info(f"Starting hyperparameter optimization with {args.n_trials} trials")
        best_params = model_trainer.hyperparameter_optimization(
            train_dataloader, val_dataloader, n_trials=args.n_trials
        )
        
        # Train with best hyperparameters
        logger.info("Training model with best hyperparameters")
        model = model_trainer.train_with_best_params(
            best_params, train_dataloader, val_dataloader
        )
    else:
        # Regular training
        logger.info("Starting regular model training")
        model = model_trainer.train(train_dataloader, val_dataloader)
    
    logger.info("Training completed")
    
    # Save training metadata
    metadata = {
        "exp_dir": exp_dir,
        "config_path": os.path.join(exp_dir, "config.yaml"),
        "data_path": config["data"]["csv_file_path"],
        "preprocessor_path": os.path.join(exp_dir, "preprocessor"),
        "hyperparameter_optimization": args.hp_opt,
    }
    
    with open(os.path.join(exp_dir, "training_metadata.yaml"), "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Saved training metadata to {os.path.join(exp_dir, 'training_metadata.yaml')}")
    logger.info(f"Training complete. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()