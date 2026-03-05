"""Visualization utilities for counterfactual explanations."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class CounterfactualVisualizer:
    """Visualizer for counterfactual explanations."""
    
    def __init__(self, feature_names: List[str], target_names: List[str]):
        """Initialize visualizer.
        
        Args:
            feature_names: List of feature names.
            target_names: List of target class names.
        """
        self.feature_names = feature_names
        self.target_names = target_names
        plt.style.use('seaborn-v0_8')
    
    def plot_feature_comparison(
        self, 
        original: np.ndarray, 
        counterfactual: np.ndarray,
        title: str = "Original vs Counterfactual",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature comparison between original and counterfactual.
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original instance
        bars1 = ax1.bar(self.feature_names, original, color='skyblue', alpha=0.7)
        ax1.set_title(f"{title} - Original")
        ax1.set_ylabel("Feature Value")
        ax1.tick_params(axis='x', rotation=45)
        
        # Counterfactual instance
        bars2 = ax2.bar(self.feature_names, counterfactual, color='lightcoral', alpha=0.7)
        ax2.set_title(f"{title} - Counterfactual")
        ax2.set_ylabel("Feature Value")
        ax2.tick_params(axis='x', rotation=45)
        
        # Highlight differences
        changes = counterfactual - original
        for i, (bar1, bar2, change) in enumerate(zip(bars1, bars2, changes)):
            if abs(change) > 1e-6:
                bar1.set_edgecolor('red')
                bar1.set_linewidth(2)
                bar2.set_edgecolor('red')
                bar2.set_linewidth(2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_changes(
        self, 
        original: np.ndarray, 
        counterfactual: np.ndarray,
        title: str = "Feature Changes",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature changes as a bar chart.
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            title: Plot title.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        changes = counterfactual - original
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if change < 0 else 'green' for change in changes]
        bars = ax.bar(self.feature_names, changes, color=colors, alpha=0.7)
        
        ax.set_title(title)
        ax.set_ylabel("Change in Feature Value")
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, change in zip(bars, changes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{change:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_comparison(
        self, 
        original: np.ndarray, 
        counterfactual: np.ndarray,
        title: str = "Interactive Feature Comparison"
    ) -> go.Figure:
        """Create interactive plotly comparison.
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            title: Plot title.
            
        Returns:
            Plotly figure.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Original", "Counterfactual"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Original
        fig.add_trace(
            go.Bar(
                x=self.feature_names,
                y=original,
                name="Original",
                marker_color='skyblue',
                text=[f"{val:.3f}" for val in original],
                textposition='auto',
            ),
            row=1, col=1
        )
        
        # Counterfactual
        fig.add_trace(
            go.Bar(
                x=self.feature_names,
                y=counterfactual,
                name="Counterfactual",
                marker_color='lightcoral',
                text=[f"{val:.3f}" for val in counterfactual],
                textposition='auto',
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=500,
            width=1000
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Feature Value")
        
        return fig
    
    def plot_metrics_comparison(
        self, 
        metrics_data: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of metrics across methods.
        
        Args:
            metrics_data: Dictionary mapping method names to metrics.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        methods = list(metrics_data.keys())
        metrics = ['proximity', 'sparsity', 'validity', 'feasibility']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [metrics_data[method].get(metric, 0) for method in methods]
            
            bars = axes[i].bar(methods, values, alpha=0.7)
            axes[i].set_title(f"{metric.title()} Score")
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_leaderboard(
        self, 
        leaderboard_df: pd.DataFrame,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot leaderboard as a horizontal bar chart.
        
        Args:
            leaderboard_df: DataFrame with leaderboard data.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by composite score
        df_sorted = leaderboard_df.sort_values('composite_score', ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        
        bars = ax.barh(y_pos, df_sorted['composite_score'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['method'])
        ax.set_xlabel('Composite Score')
        ax.set_title('Counterfactual Methods Leaderboard')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, df_sorted['composite_score'])):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
