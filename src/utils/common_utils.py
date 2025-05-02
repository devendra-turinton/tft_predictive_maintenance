import os
import logging
import yaml
import torch
import numpy as np
import random
import pandas as pd
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from datetime import datetime
import json

def setup_logging(log_dir: str = "logs", level=logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tft_pm_{timestamp}.log")
    
    # Set up logging configuration
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed}")

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config

def save_config(config: Dict, output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logging.info(f"Saved configuration to {output_path}")

def create_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Create a timestamped directory for the current experiment.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to the created experiment directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamp for experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    # Create experiment directory and subdirectories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    
    logging.info(f"Created experiment directory: {exp_dir}")
    return exp_dir

def update_config_paths(config: Dict, exp_dir: str) -> Dict:
    """
    Update configuration paths based on experiment directory.
    
    Args:
        config: Configuration dictionary
        exp_dir: Experiment directory
        
    Returns:
        Updated configuration dictionary
    """
    # Update training paths
    if "training" in config:
        config["training"]["log_dir"] = os.path.join(exp_dir, "logs")
        config["training"]["checkpoint_dir"] = os.path.join(exp_dir, "checkpoints")
    
    # Update evaluation paths
    if "evaluation" in config:
        config["evaluation"]["output_dir"] = os.path.join(exp_dir, "results")
        config["evaluation"]["plot_dir"] = os.path.join(exp_dir, "plots")
    
    logging.info(f"Updated configuration paths for experiment: {exp_dir}")
    return config

def save_predictions(
    predictions: pd.DataFrame,
    output_path: str,
    format: str = "csv"
) -> None:
    """
    Save predictions to file.
    
    Args:
        predictions: DataFrame with predictions
        output_path: Path to save predictions
        format: File format ('csv' or 'parquet')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == "csv":
        predictions.to_csv(output_path, index=False)
    elif format.lower() == "parquet":
        predictions.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Saved predictions to {output_path}")

def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save metrics
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Saved metrics to {output_path}")

def calculate_system_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate system statistics from data.
    
    Args:
        df: DataFrame with data
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Count number of unique machines
    if "machine_id" in df.columns:
        stats["num_machines"] = df["machine_id"].nunique()
    
    # Count total number of time steps
    if "timestamp" in df.columns:
        stats["start_date"] = df["timestamp"].min()
        stats["end_date"] = df["timestamp"].max()
        stats["total_time_span"] = (stats["end_date"] - stats["start_date"]).total_seconds() / (3600 * 24)  # in days
    
    # Count number of failures
    if "time_to_failure" in df.columns:
        # Assuming failures are defined as time_to_failure <= threshold
        threshold = 0.5  # Adjust based on your definition
        stats["num_failures"] = (df["time_to_failure"] <= threshold).sum()
        stats["failure_rate"] = stats["num_failures"] / len(df) if len(df) > 0 else 0
    
    # Data density statistics
    stats["total_records"] = len(df)
    stats["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    logging.info(f"Calculated system statistics")
    return stats

def check_gpu_availability() -> bool:
    """
    Check if GPU is available.
    
    Returns:
        True if GPU is available, False otherwise
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        logging.info(f"GPU is available: {device_count} device(s), first device: {device_name}")
        return True
    else:
        logging.info("No GPU available, using CPU")
        return False

def create_sample_data(
    num_machines: int = 5,
    num_days: int = 30,
    hours_per_day: int = 24,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create sample data for testing.
    
    Args:
        num_machines: Number of machines
        num_days: Number of days of data
        hours_per_day: Hours of data per day
        random_seed: Random seed
        output_path: Path to save sample data
        
    Returns:
        DataFrame with sample data
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Create data
    data = []
    start_date = datetime(2023, 1, 1)
    
    for machine_id in range(1, num_machines + 1):
        # Static features
        machine_type = np.random.choice(["Type A", "Type B", "Type C"])
        installation_date = datetime(2020, np.random.randint(1, 13), np.random.randint(1, 29))
        max_capacity = np.random.uniform(80, 120)
        facility = np.random.choice(["North", "South", "East", "West"])
        
        # Generate sensor patterns
        base_vibration = np.random.uniform(0.5, 1.5)
        base_temperature = np.random.uniform(50, 70)
        base_pressure = np.random.uniform(80, 120)
        base_current = np.random.uniform(4, 6)
        base_voltage = np.random.uniform(220, 240)
        base_rpm = np.random.uniform(1000, 2000)
        
        # Time-to-failure pattern (decreasing over time with some noise)
        # Each machine will have 1-2 failure events
        num_failures = np.random.randint(1, 3)
        failure_times = np.sort(np.random.choice(
            np.arange(num_days * hours_per_day), 
            size=num_failures, 
            replace=False
        ))
        
        for hour in range(num_days * hours_per_day):
            timestamp = start_date + pd.Timedelta(hours=hour)
            
            # Time index
            time_idx = hour
            
            # Calculate time to failure
            time_to_failure = min([ft - hour for ft in failure_times if ft > hour] + [1000])
            
            # Generate time-varying features
            operational_mode = np.random.choice(["Normal", "High", "Low"])
            maintenance_status = "Maintained" if np.random.random() < 0.05 else "Normal"
            scheduled_load = np.random.uniform(50, 100)
            ambient_temperature = np.random.uniform(15, 35)
            
            # Sensor readings - deteriorate as time_to_failure decreases
            deterioration = 1 + max(0, (30 - time_to_failure) / 30) * np.random.uniform(0.5, 1.5)
            vibration = base_vibration * deterioration * (1 + np.random.normal(0, 0.1))
            temperature = base_temperature * deterioration * (1 + np.random.normal(0, 0.05))
            pressure = base_pressure * (1 - (deterioration - 1) * 0.5) * (1 + np.random.normal(0, 0.08))
            current = base_current * deterioration * (1 + np.random.normal(0, 0.07))
            voltage = base_voltage * (1 - (deterioration - 1) * 0.2) * (1 + np.random.normal(0, 0.03))
            rpm = base_rpm * (1 - (deterioration - 1) * 0.3) * (1 + np.random.normal(0, 0.06))
            
            # Add row to data
            data.append({
                "machine_id": machine_id,
                "timestamp": timestamp,
                "time_idx": time_idx,
                "machine_type": machine_type,
                "installation_date": installation_date,
                "max_capacity": max_capacity,
                "facility": facility,
                "operational_mode": operational_mode,
                "maintenance_status": maintenance_status,
                "scheduled_load": scheduled_load,
                "ambient_temperature": ambient_temperature,
                "vibration": vibration,
                "temperature": temperature,
                "pressure": pressure,
                "current": current,
                "voltage": voltage,
                "rpm": rpm,
                "time_to_failure": time_to_failure
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to file if output_path is provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Saved sample data to {output_path}")
    
    return df