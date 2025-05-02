import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Tuple, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class to handle loading and initial processing of sensor data.
    """
    def __init__(self, config: Dict):
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            config: Dictionary containing data configuration
        """
        self.config = config
        self.data_path = config["csv_file_path"]
        self.time_col = config["time_column"]
        self.id_col = config["id_column"]
        self.target_col = config["target_column"]
        self.time_idx = config["time_idx"]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file and perform basic validation.
        
        Returns:
            DataFrame: Loaded and validated sensor data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            # Load the data
            df = pd.read_csv(self.data_path)
            
            # Validate required columns exist
            required_cols = [self.time_col, self.id_col, self.target_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert timestamp to datetime if it's not already
            if pd.api.types.is_string_dtype(df[self.time_col]):
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                
            # Create time index if it doesn't exist
            if self.time_idx not in df.columns:
                # Group by ID and create a time index for each group
                df = self._create_time_idx(df)
                
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def _create_time_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a time index column for each group.
        
        Args:
            df: DataFrame without time index
            
        Returns:
            DataFrame with added time index
        """
        # Sort by ID and timestamp
        df = df.sort_values([self.id_col, self.time_col])
        
        # Create time index per group
        df[self.time_idx] = df.groupby(self.id_col).cumcount()
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation and test sets based on config ratios.
        Uses a time-based split to prevent data leakage.
        
        Args:
            df: Complete DataFrame to split
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train, validation, and test sets")
        
        # Get unique machine IDs
        machine_ids = df[self.id_col].unique()
        
        train_ratio = self.config["train_ratio"]
        val_ratio = self.config["val_ratio"]
        
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        # Split each machine's data separately to ensure balanced representation
        for machine_id in machine_ids:
            machine_data = df[df[self.id_col] == machine_id].sort_values(self.time_col)
            
            # Calculate split indices
            n = len(machine_data)
            train_idx = int(n * train_ratio)
            val_idx = train_idx + int(n * val_ratio)
            
            # Split the data
            train_dfs.append(machine_data.iloc[:train_idx])
            val_dfs.append(machine_data.iloc[train_idx:val_idx])
            test_dfs.append(machine_data.iloc[val_idx:])
        
        # Combine the splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        logger.info(f"Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split the data in one function call.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = self.load_data()
        return self.split_data(df)