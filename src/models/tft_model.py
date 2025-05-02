import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss

logger = logging.getLogger(__name__)

class PredictiveMaintenanceTFT(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Temporal Fusion Transformer model
    for predictive maintenance.
    """
    def __init__(self, config: Dict, dataset_parameters: Dict):
        """
        Initialize the TFT model.
        
        Args:
            config: Model configuration
            dataset_parameters: Parameters from the TimeSeriesDataSet
        """
        super().__init__()
        self.config = config
        self.dataset_params = dataset_parameters
        self.save_hyperparameters(
            {**config, "dataset_parameters": {k: str(v) for k, v in dataset_parameters.items()}}
        )
        
        # Create loss function - Quantile Loss for probabilistic forecasting
        self.loss = QuantileLoss()
        
        # Create TFT model
        self.model = self._build_model()
        
        logger.info("Initialized TFT model")
        
    def _build_model(self) -> TemporalFusionTransformer:
        """
        Build the TFT model with configuration parameters.
        
        Returns:
            TemporalFusionTransformer model
        """
        return TemporalFusionTransformer(
            # Dataset parameters
            **self.dataset_params,
            
            # Model parameters from config
            hidden_size=self.config.get("hidden_size", 64),
            lstm_layers=self.config.get("lstm_layers", 2),
            attention_head_size=self.config.get("attention_head_size", 4),
            dropout=self.config.get("dropout", 0.1),
            hidden_continuous_size=self.config.get("hidden_continuous_size", 32),
            
            # Loss function
            loss=self.loss,
            
            # Optimization parameters
            learning_rate=self.config.get("learning_rate", 0.001),
            reduce_on_plateau_patience=self.config.get("reduce_on_plateau_patience", 3),
            
            # Log parameters
            log_interval=self.config.get("log_interval", 10),
            log_val_interval=self.config.get("log_val_interval", 1),
        )
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input dictionary
            
        Returns:
            Model outputs
        """
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a training step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Loss tensor
        """
        # Pass batch to model
        output = self.model.training_step(batch, batch_idx)
        
        # Log detailed metrics if needed
        if batch_idx % self.config.get("log_interval", 10) == 0:
            self._log_additional_metrics(batch, output, "train")
            
        return output
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a validation step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with validation metrics
        """
        # Pass batch to model
        output = self.model.validation_step(batch, batch_idx)
        
        # Log detailed metrics if needed
        if batch_idx % self.config.get("log_val_interval", 1) == 0:
            self._log_additional_metrics(batch, output, "val")
            
        return output
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform a test step.
        
        Args:
            batch: Batch of data
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with test metrics
        """
        # Pass batch to model
        output = self.model.validation_step(batch, batch_idx)  # TFT uses validation_step for testing too
        
        # Log detailed metrics
        self._log_additional_metrics(batch, output, "test")
            
        return output
    
    def _log_additional_metrics(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor], stage: str) -> None:
        """
        Log additional metrics beyond the default ones.
        
        Args:
            batch: Batch of data
            output: Model output
            stage: 'train', 'val', or 'test'
        """
        # Calculate MAE
        if 'prediction' in output and 'target' in batch:
            # Get the median prediction (0.5 quantile)
            if isinstance(output['prediction'], dict) and 0.5 in output['prediction']:
                prediction = output['prediction'][0.5]
            else:
                prediction = output['prediction']
                
            # Calculate MAE
            mae = torch.mean(torch.abs(prediction - batch['target']).mean(dim=1))
            self.log(f"{stage}_mae", mae, prog_bar=True)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and LR schedulers.
        
        Returns:
            Dictionary with optimizer configuration
        """
        # The TFT model from PyTorch Forecasting handles this internally,
        # but we can override it here if needed
        return self.model.configure_optimizers()
    
    def predict(
        self, 
        data: Dict[str, torch.Tensor],
        return_quantiles: List[float] = [0.1, 0.5, 0.9],
        mode: str = "prediction"
    ) -> Union[torch.Tensor, Dict[float, torch.Tensor]]:
        """
        Make predictions with the model.
        
        Args:
            data: Input data
            return_quantiles: List of quantiles to return
            mode: 'prediction' or 'raw' to get raw outputs
            
        Returns:
            Model predictions or raw outputs
        """
        return self.model.predict(data, return_quantiles=return_quantiles, mode=mode)
    
    def get_attention_weights(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (encoder_attention, decoder_attention)
        """
        return self.model.get_attention_weights(data)
    
    def get_interpretation(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Get feature importances and other interpretation metrics.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with interpretation metrics
        """
        interp = self.model.interpret_output(
            self.model.predict(data, mode="raw"),
            reduction="mean"  # or can use 'none' for no aggregation
        )
        
        return interp