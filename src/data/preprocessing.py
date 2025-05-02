import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
import os

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles preprocessing of time series data for TFT model including:
    - Missing value imputation
    - Feature scaling
    - Feature engineering
    - Handling categorical variables
    """
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config
        self.static_cats = config.get("static_categoricals", [])
        self.static_reals = config.get("static_reals", [])
        self.time_varying_known_cats = config.get("time_varying_known_categoricals", [])
        self.time_varying_known_reals = config.get("time_varying_known_reals", [])
        self.time_varying_unknown_cats = config.get("time_varying_unknown_categoricals", [])
        self.time_varying_unknown_reals = config.get("time_varying_unknown_reals", [])
        
        self.target_col = config["target_column"]
        self.id_col = config["id_column"]
        self.time_col = config["time_column"]
        
        # Initialize scalers and imputers
        self.scalers = {}
        self.imputers = {}
        
    def fit_transform(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Fit and transform the data. If is_training is False, only transform is applied.
        
        Args:
            df: DataFrame to preprocess
            is_training: Whether this is training data (to fit scalers/imputers)
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info(f"Preprocessing {'training' if is_training else 'evaluation'} data")
        
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df, is_training)
        
        # Apply feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # Scale numerical features
        processed_df = self._scale_features(processed_df, is_training)
        
        # Handle categorical features
        processed_df = self._encode_categorical_features(processed_df, is_training)
        
        logger.info(f"Preprocessing complete. Output shape: {processed_df.shape}")
        return processed_df
        
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Impute missing values in the dataset.
        
        Args:
            df: DataFrame with potentially missing values
            is_training: Whether to fit imputers on this data
            
        Returns:
            DataFrame with imputed values
        """
        # Get all real-valued columns that might have missing values
        real_cols = (
            self.static_reals + 
            self.time_varying_known_reals + 
            self.time_varying_unknown_reals
        )
        
        # Handle each column
        for col in real_cols:
            if col in df.columns:
                if is_training:
                    # For training data, fit a new imputer
                    imputer = SimpleImputer(strategy='median')
                    df[col] = imputer.fit_transform(df[[col]])
                    # Save the imputer
                    self.imputers[col] = imputer
                else:
                    # For non-training data, use pre-fitted imputer
                    if col in self.imputers:
                        df[col] = self.imputers[col].transform(df[[col]])
                    else:
                        # If no imputer exists, use median of this data
                        df[col] = df[col].fillna(df[col].median())
                        
        # For categorical columns, fill with mode
        cat_cols = (
            self.static_cats + 
            self.time_varying_known_cats + 
            self.time_varying_unknown_cats
        )
        
        for col in cat_cols:
            if col in df.columns:
                # For categorical data, fill with most common value
                most_common = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(most_common)
                
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features that might help the model.
        
        Args:
            df: DataFrame to engineer features for
            
        Returns:
            DataFrame with additional engineered features
        """
        # Extract datetime components if timestamp is present
        if self.time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            # Extract time-based features
            df['hour'] = df[self.time_col].dt.hour
            df['day'] = df[self.time_col].dt.day
            df['month'] = df[self.time_col].dt.month
            df['day_of_week'] = df[self.time_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Add these as known time-varying features
            self.time_varying_known_reals.extend(['hour', 'day', 'month', 'day_of_week', 'is_weekend'])
        
        # Engineering for sensor data - create rolling statistics for unknown real features
        if len(self.time_varying_unknown_reals) > 0:
            # Group by machine ID to calculate rolling stats per machine
            grouped = df.groupby(self.id_col)
            
            # Calculate rolling statistics for each sensor
            for col in self.time_varying_unknown_reals:
                if col in df.columns:
                    # Rolling mean with window of 24 time steps
                    df[f'{col}_rolling_mean_24'] = grouped[col].transform(
                        lambda x: x.rolling(window=24, min_periods=1).mean()
                    )
                    
                    # Rolling standard deviation
                    df[f'{col}_rolling_std_24'] = grouped[col].transform(
                        lambda x: x.rolling(window=24, min_periods=1).std()
                    )
                    
                    # Rolling max
                    df[f'{col}_rolling_max_24'] = grouped[col].transform(
                        lambda x: x.rolling(window=24, min_periods=1).max()
                    )
                    
                    # Rate of change (first derivative approximation)
                    df[f'{col}_rate_of_change'] = grouped[col].transform(
                        lambda x: x.diff() / x.shift(1)
                    ).fillna(0)
                    
                    # Add these to unknown real features
                    self.time_varying_unknown_reals.extend([
                        f'{col}_rolling_mean_24', 
                        f'{col}_rolling_std_24',
                        f'{col}_rolling_max_24',
                        f'{col}_rate_of_change'
                    ])
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame with features to scale
            is_training: Whether to fit scalers on this data
            
        Returns:
            DataFrame with scaled features
        """
        # Get all real-valued columns for scaling
        real_cols = (
            self.static_reals + 
            self.time_varying_known_reals + 
            self.time_varying_unknown_reals
        )
        
        # Scale the target column separately with MinMaxScaler to [0,1]
        if self.target_col in df.columns:
            if is_training:
                target_scaler = MinMaxScaler(feature_range=(0, 1))
                df[self.target_col] = target_scaler.fit_transform(df[[self.target_col]])
                self.scalers[self.target_col] = target_scaler
            else:
                if self.target_col in self.scalers:
                    df[self.target_col] = self.scalers[self.target_col].transform(df[[self.target_col]])
        
        # Scale other real-valued columns with StandardScaler
        for col in real_cols:
            if col in df.columns:
                if is_training:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        df[col] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Encode categorical features - for TFT we don't need to one-hot encode
        as the model handles categoricals directly.
        
        Args:
            df: DataFrame with categorical features
            is_training: Whether this is training data
            
        Returns:
            DataFrame with encoded categoricals
        """
        # Get all categorical columns
        cat_cols = (
            self.static_cats + 
            self.time_varying_known_cats + 
            self.time_varying_unknown_cats
        )
        
        # Initialize cat_codes attribute if it doesn't exist
        if not hasattr(self, 'cat_codes'):
            self.cat_codes = {}
        
        # For each categorical column, ensure it's encoded as a category type
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
                
                # If training, store category codes for future use
                if is_training:
                    self.cat_codes[col] = dict(enumerate(df[col].cat.categories))
        
        return df
    
    def save_preprocessor(self, output_dir: str) -> None:
        """
        Save the fitted preprocessor for later use.
        
        Args:
            output_dir: Directory to save preprocessor to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(output_dir, 'scalers.pkl'))
        
        # Save imputers
        joblib.dump(self.imputers, os.path.join(output_dir, 'imputers.pkl'))
        
        # Save categorical codes if they exist
        if hasattr(self, 'cat_codes'):
            joblib.dump(self.cat_codes, os.path.join(output_dir, 'cat_codes.pkl'))
        
        # Save configuration
        joblib.dump({
            'static_cats': self.static_cats,
            'static_reals': self.static_reals,
            'time_varying_known_cats': self.time_varying_known_cats,
            'time_varying_known_reals': self.time_varying_known_reals,
            'time_varying_unknown_cats': self.time_varying_unknown_cats,
            'time_varying_unknown_reals': self.time_varying_unknown_reals,
        }, os.path.join(output_dir, 'feature_config.pkl'))
        
        logger.info(f"Saved preprocessor to {output_dir}")
    
    @classmethod
    def load_preprocessor(cls, input_dir: str, config: Dict) -> 'DataPreprocessor':
        """
        Load a previously saved preprocessor.
        
        Args:
            input_dir: Directory containing saved preprocessor
            config: Base configuration dictionary
            
        Returns:
            Loaded DataPreprocessor instance
        """
        preprocessor = cls(config)
        
        # Load scalers
        preprocessor.scalers = joblib.load(os.path.join(input_dir, 'scalers.pkl'))
        
        # Load imputers
        preprocessor.imputers = joblib.load(os.path.join(input_dir, 'imputers.pkl'))
        
        # Load categorical codes if they exist
        cat_codes_path = os.path.join(input_dir, 'cat_codes.pkl')
        if os.path.exists(cat_codes_path):
            preprocessor.cat_codes = joblib.load(cat_codes_path)
        
        # Load feature configuration
        feature_config = joblib.load(os.path.join(input_dir, 'feature_config.pkl'))
        preprocessor.static_cats = feature_config['static_cats']
        preprocessor.static_reals = feature_config['static_reals']
        preprocessor.time_varying_known_cats = feature_config['time_varying_known_cats']
        preprocessor.time_varying_known_reals = feature_config['time_varying_known_reals']
        preprocessor.time_varying_unknown_cats = feature_config['time_varying_unknown_cats']
        preprocessor.time_varying_unknown_reals = feature_config['time_varying_unknown_reals']
        
        logger.info(f"Loaded preprocessor from {input_dir}")
        return preprocessor