import pytorch_lightning as pl
import torch
import os
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from ..models.tft_model import PredictiveMaintenanceTFT
from ..models.model_utils import setup_trainer
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_forecasting.metrics import QuantileLoss

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles model training, including hyperparameter optimization if required.
    """
    def __init__(self, config: Dict, dataset_params: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            dataset_params: Dataset parameters
        """
        self.config = config
        self.dataset_params = dataset_params
        self.train_config = config.get("training", {})
        self.model_config = config.get("model", {})
        
        # Set random seeds for reproducibility
        seed = self.train_config.get("seed", 42)
        pl.seed_everything(seed)
        
        # Create output directories
        os.makedirs(self.train_config.get("log_dir", "logs"), exist_ok=True)
        os.makedirs(self.train_config.get("checkpoint_dir", "checkpoints"), exist_ok=True)
        
        logger.info("Initialized ModelTrainer")
        
    def train(
        self, 
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> pl.LightningModule:
        """
        Train the TFT model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Trained model
        """
        logger.info("Starting model training")
        
        # Create model
        model = PredictiveMaintenanceTFT(
            config=self.model_config,
            dataset_parameters=self.dataset_params
        )
        
        # Set up trainer
        trainer = setup_trainer(self.train_config)
        
        # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        logger.info("Model training completed")
        
        return model
    
    def hyperparameter_optimization(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        n_trials: int = 20
    ) -> Dict:
        """
        Perform hyperparameter optimization using Optuna.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial: optuna.Trial) -> float:
            """
            Objective function for Optuna optimization.
            
            Args:
                trial: Optuna trial
                
            Returns:
                Validation loss
            """
            # Define hyperparameters to tune
            hp_config = {
                "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
                "lstm_layers": trial.suggest_int("lstm_layers", 1, 3),
                "attention_head_size": trial.suggest_int("attention_head_size", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.3),
                "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32, 64]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            }
            
            # Update model config with trial hyperparameters
            model_config = {**self.model_config, **hp_config}
            
            # Create model with trial hyperparameters
            model = PredictiveMaintenanceTFT(
                config=model_config,
                dataset_parameters=self.dataset_params
            )
            
            # Set up trainer with pruning callback
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
            train_config = {**self.train_config, "max_epochs": 10}  # Limit epochs for faster trials
            trainer = setup_trainer(train_config)
            trainer.callbacks.append(pruning_callback)
            
            # Train model
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            # Return best validation loss
            return trainer.callback_metrics["val_loss"].item()
        
        # Create Optuna study
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Hyperparameter optimization completed")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_trial.value}")
        logger.info(f"Best hyperparameters: {study.best_trial.params}")
        
        # Save best hyperparameters
        best_params = study.best_trial.params
        output_path = os.path.join(self.train_config.get("log_dir", "logs"), "best_params.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        
        return best_params
    
    def fine_tune(
        self,
        model_path: str,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        epochs: Optional[int] = None
    ) -> pl.LightningModule:
        """
        Fine-tune a pre-trained model.
        
        Args:
            model_path: Path to pre-trained model checkpoint
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of fine-tuning epochs (uses config if None)
            
        Returns:
            Fine-tuned model
        """
        logger.info(f"Starting model fine-tuning from {model_path}")
        
        # Load model
        model = PredictiveMaintenanceTFT.load_from_checkpoint(
            model_path,
            config=self.model_config,
            dataset_parameters=self.dataset_params
        )
        
        # Modify learning rate for fine-tuning
        model.model.hparams.learning_rate = self.model_config.get("fine_tuning_lr", 1e-4)
        
        # Set up trainer
        train_config = {**self.train_config}
        if epochs is not None:
            train_config["max_epochs"] = epochs
        
        trainer = setup_trainer(train_config)
        
        # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        logger.info("Model fine-tuning completed")
        
        return model
    
    def train_with_best_params(
        self,
        best_params: Dict,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader
    ) -> pl.LightningModule:
        """
        Train model with best hyperparameters from optimization.
        
        Args:
            best_params: Best hyperparameters from optimization
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            
        Returns:
            Trained model with best hyperparameters
        """
        logger.info("Training model with best hyperparameters")
        
        # Update model config with best hyperparameters
        model_config = {**self.model_config, **best_params}
        
        # Create model with best hyperparameters
        model = PredictiveMaintenanceTFT(
            config=model_config,
            dataset_parameters=self.dataset_params
        )
        
        # Set up trainer
        trainer = setup_trainer(self.train_config)
        
        # Train the model
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        logger.info("Model training with best hyperparameters completed")
        
        return model