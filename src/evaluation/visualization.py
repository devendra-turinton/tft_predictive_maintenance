import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PredictiveMaintenanceVisualizer:
    """
    Class for creating visualizations for predictive maintenance results.
    """
    def __init__(self, config: Dict):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self.num_samples = config.get("visualization_samples", 5)
        self.confidence_intervals = config.get("confidence_intervals", [50, 90])
        
        # Set default plot style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        
    def plot_predictions(
        self,
        predictions: Dict[float, np.ndarray],
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        machine_ids: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> List[Figure]:
        """
        Plot predictions vs. actual values with confidence intervals.
        
        Args:
            predictions: Dictionary of predictions for different quantiles
            targets: Actual target values
            timestamps: Timestamps for each prediction
            machine_ids: Optional machine IDs for each prediction
            output_dir: Directory to save plots to
            
        Returns:
            List of created figure objects
        """
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        figures = []
        
        # If machine_ids is provided, plot separately for each machine
        if machine_ids is not None:
            unique_machines = np.unique(machine_ids)
            
            # Limit to num_samples machines if there are too many
            if len(unique_machines) > self.num_samples:
                unique_machines = unique_machines[:self.num_samples]
                logger.info(f"Limiting to {self.num_samples} machines for visualization")
            
            for machine_id in unique_machines:
                # Filter data for this machine
                mask = machine_ids == machine_id
                machine_targets = targets[mask]
                machine_timestamps = timestamps[mask]
                machine_predictions = {q: pred[mask] for q, pred in predictions.items()}
                
                # Create and save plot
                fig = self._create_prediction_plot(
                    machine_predictions, machine_targets, machine_timestamps,
                    title=f"Machine {machine_id} - Predictions vs. Actual"
                )
                
                figures.append(fig)
                
                if output_dir is not None:
                    fig_path = os.path.join(output_dir, f"predictions_machine_{machine_id}.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
        else:
            # Plot all data together
            fig = self._create_prediction_plot(
                predictions, targets, timestamps,
                title="Predictions vs. Actual"
            )
            
            figures.append(fig)
            
            if output_dir is not None:
                fig_path = os.path.join(output_dir, "predictions_all.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
        
        return figures
    
    def _create_prediction_plot(
        self,
        predictions: Dict[float, np.ndarray],
        targets: np.ndarray,
        timestamps: pd.DatetimeIndex,
        title: str = "Predictions vs. Actual"
    ) -> Figure:
        """
        Create a single prediction plot.
        
        Args:
            predictions: Dictionary of predictions for different quantiles
            targets: Actual target values
            timestamps: Timestamps for each prediction
            title: Plot title
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual values
        ax.plot(timestamps, targets, 'k.-', label='Actual', alpha=0.7)
        
        # Get median prediction (q=0.5)
        if 0.5 in predictions:
            median_predictions = predictions[0.5]
            ax.plot(timestamps, median_predictions, 'r.-', label='Median Prediction', alpha=0.9)
        
        # Plot confidence intervals
        for ci in self.confidence_intervals:
            lower_q = (100 - ci) / 200
            upper_q = (100 + ci) / 200
            
            # Check if we have these quantiles
            if lower_q in predictions and upper_q in predictions:
                lower_bound = predictions[lower_q]
                upper_bound = predictions[upper_q]
                
                ax.fill_between(
                    timestamps, lower_bound, upper_bound,
                    alpha=0.2, label=f"{ci}% Confidence Interval"
                )
        
        # Set plot properties
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Time to Failure')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        fig.autofmt_xdate()
        
        # Add threshold line if applicable
        if "failure_threshold" in self.config:
            threshold = self.config["failure_threshold"]
            ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                      label=f'Failure Threshold ({threshold})')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, np.ndarray],
        output_dir: Optional[str] = None
    ) -> Figure:
        """
        Plot feature importance from TFT model.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            output_dir: Directory to save plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Convert feature importance to DataFrame for easier plotting
        importance_df = pd.DataFrame()
        
        # Static features
        if "static_features" in feature_importance:
            static_features = feature_importance["static_features"]
            importance_df = pd.concat([
                importance_df,
                pd.DataFrame({
                    "Feature": static_features.index,
                    "Importance": static_features.values,
                    "Type": "Static"
                })
            ], ignore_index=True)
        
        # Encoder features
        if "encoder_features" in feature_importance:
            encoder_features = feature_importance["encoder_features"]
            importance_df = pd.concat([
                importance_df,
                pd.DataFrame({
                    "Feature": encoder_features.index,
                    "Importance": encoder_features.values,
                    "Type": "Historical"
                })
            ], ignore_index=True)
            
        # Decoder features
        if "decoder_features" in feature_importance:
            decoder_features = feature_importance["decoder_features"]
            importance_df = pd.concat([
                importance_df,
                pd.DataFrame({
                    "Feature": decoder_features.index,
                    "Importance": decoder_features.values,
                    "Type": "Future"
                })
            ], ignore_index=True)
        
        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot feature importance
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            hue="Type",
            ax=ax
        )
        
        # Set plot properties
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, "feature_importance.png")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        return fig
    
    def plot_attention_patterns(
        self,
        attention_weights: Dict[str, np.ndarray],
        timestamps: pd.DatetimeIndex,
        output_dir: Optional[str] = None
    ) -> List[Figure]:
        """
        Plot attention patterns from TFT model.
        
        Args:
            attention_weights: Dictionary of attention weights
            timestamps: Timestamps for reference
            output_dir: Directory to save plots to
            
        Returns:
            List of Matplotlib Figure objects
        """
        figures = []
        
        # Encoder-decoder attention
        if "encoder_decoder_attention" in attention_weights:
            enc_dec_attention = attention_weights["encoder_decoder_attention"]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot heatmap
            sns.heatmap(
                enc_dec_attention,
                cmap="viridis",
                ax=ax
            )
            
            # Set plot properties
            ax.set_title("Encoder-Decoder Attention")
            ax.set_xlabel("Encoder Time Steps")
            ax.set_ylabel("Decoder Time Steps")
            
            figures.append(fig)
            
            # Save plot if output_dir is provided
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                fig_path = os.path.join(output_dir, "encoder_decoder_attention.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
        
        # Self-attention
        if "self_attention" in attention_weights:
            self_attention = attention_weights["self_attention"]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot heatmap
            sns.heatmap(
                self_attention,
                cmap="viridis",
                ax=ax
            )
            
            # Set plot properties
            ax.set_title("Self Attention")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Time Steps")
            
            figures.append(fig)
            
            # Save plot if output_dir is provided
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                fig_path = os.path.join(output_dir, "self_attention.png")
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
        
        return figures
    
    def plot_metrics_over_time(
        self,
        metrics_history: Dict[str, List[float]],
        output_dir: Optional[str] = None
    ) -> Figure:
        """
        Plot metrics over training time.
        
        Args:
            metrics_history: Dictionary of metric name to list of values over time
            output_dir: Directory to save plot to
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each metric
        for metric_name, values in metrics_history.items():
            ax.plot(values, label=metric_name)
        
        # Set plot properties
        ax.set_title("Metrics Over Training")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        fig.tight_layout()
        
        # Save plot if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, "metrics_history.png")
            fig.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        
        return fig