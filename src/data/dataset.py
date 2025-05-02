import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import torch
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer, EncoderNormalizer

logger = logging.getLogger(__name__)

class PredictiveMaintenanceDataset:
    """
    Creates and manages TimeSeriesDataSet objects for TFT model.
    """
    def __init__(self, config: Dict):
        """
        Initialize dataset creator with configuration.
        
        Args:
            config: Dictionary containing dataset configuration
        """
        self.config = config
        self.max_prediction_length = config["max_prediction_length"]
        self.max_encoder_length = config["max_encoder_length"]
        self.min_encoder_length = config.get("min_encoder_length", config["max_prediction_length"])
        self.batch_size = config["batch_size"]
        
        self.id_col = config["id_column"]
        self.time_idx = config["time_idx"]
        self.target_col = config["target_column"]
        
        # Feature groups from config
        self.static_categoricals = config.get("static_categoricals", [])
        self.static_reals = config.get("static_reals", [])
        self.time_varying_known_categoricals = config.get("time_varying_known_categoricals", [])
        self.time_varying_known_reals = config.get("time_varying_known_reals", [])
        self.time_varying_unknown_categoricals = config.get("time_varying_unknown_categoricals", [])
        self.time_varying_unknown_reals = config.get("time_varying_unknown_reals", [])
        
        # Dataset objects
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        
    def create_datasets(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Create TimeSeriesDataSet objects for training, validation, and testing.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Creating TimeSeriesDataSet objects")
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            data=train_df,
            time_idx=self.time_idx,
            target=self.target_col,
            group_ids=[self.id_col],
            max_encoder_length=self.max_encoder_length,
            min_encoder_length=self.min_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_known_categoricals=self.time_varying_known_categoricals,
            time_varying_known_reals=self.time_varying_known_reals,
            time_varying_unknown_categoricals=self.time_varying_unknown_categoricals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(
                groups=[self.id_col], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Create validation dataset based on training dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, val_df, predict=False, stop_randomization=True
        )
        
        # Create test dataset based on training dataset
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, test_df, predict=False, stop_randomization=True
        )
        
        logger.info(f"Created datasets - train: {len(self.training_dataset)}, "
                   f"val: {len(self.validation_dataset)}, test: {len(self.test_dataset)}")
        
        return self.training_dataset, self.validation_dataset, self.test_dataset
    
    def create_data_loaders(
        self, 
        train_dataset: Optional[TimeSeriesDataSet] = None,
        val_dataset: Optional[TimeSeriesDataSet] = None,
        test_dataset: Optional[TimeSeriesDataSet] = None
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Args:
            train_dataset: Optional training dataset (uses self.training_dataset if None)
            val_dataset: Optional validation dataset (uses self.validation_dataset if None)
            test_dataset: Optional test dataset (uses self.test_dataset if None)
            
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
        """
        if train_dataset is None:
            train_dataset = self.training_dataset
        if val_dataset is None:
            val_dataset = self.validation_dataset
        if test_dataset is None:
            test_dataset = self.test_dataset
            
        if not all([train_dataset, val_dataset, test_dataset]):
            raise ValueError("Datasets have not been created yet")
        
        # Create data loaders
        train_dataloader = train_dataset.to_dataloader(
            batch_size=self.batch_size, 
            train=True,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True
        )
        
        val_dataloader = val_dataset.to_dataloader(
            batch_size=self.batch_size, 
            train=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True
        )
        
        test_dataloader = test_dataset.to_dataloader(
            batch_size=self.batch_size, 
            train=False,
            num_workers=self.config.get("num_workers", 0),
            pin_memory=True
        )
        
        logger.info("Created data loaders")
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def get_parameters(self) -> Dict:
        """
        Get dataset parameters for model initialization.
        
        Returns:
            Dict of dataset parameters
        """
        if self.training_dataset is None:
            raise ValueError("Training dataset has not been created yet")
            
        return {
            "embedding_sizes": self.training_dataset.embedding_sizes,
            "continuous_columns": self.training_dataset.time_varying_reals_indices,
            "categorical_feature_names": self.training_dataset.categorical_columns,
            "embedding_paddings": self.training_dataset.embedding_paddings,
            "x_categoricals": self.training_dataset.x_categoricals,
            "x_reals": self.training_dataset.x_reals,
        }