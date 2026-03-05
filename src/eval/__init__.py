"""Evaluation metrics for counterfactual explanations."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class CounterfactualMetrics:
    """Metrics for evaluating counterfactual explanations."""
    
    @staticmethod
    def proximity_score(
        original: np.ndarray, 
        counterfactual: np.ndarray, 
        metric: str = "euclidean"
    ) -> float:
        """Calculate proximity score between original and counterfactual.
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            metric: Distance metric ('euclidean', 'manhattan', 'cosine').
            
        Returns:
            Proximity score (lower is better).
        """
        if metric == "euclidean":
            return np.linalg.norm(counterfactual - original)
        elif metric == "manhattan":
            return np.sum(np.abs(counterfactual - original))
        elif metric == "cosine":
            return 1 - np.dot(original, counterfactual) / (
                np.linalg.norm(original) * np.linalg.norm(counterfactual)
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    @staticmethod
    def sparsity_score(
        original: np.ndarray, 
        counterfactual: np.ndarray, 
        threshold: float = 1e-6
    ) -> float:
        """Calculate sparsity score (number of changed features).
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            threshold: Threshold for considering a feature as changed.
            
        Returns:
            Sparsity score (lower is better).
        """
        changes = np.abs(counterfactual - original)
        return np.sum(changes > threshold)
    
    @staticmethod
    def validity_score(
        counterfactual: np.ndarray, 
        model: Any, 
        target_class: int
    ) -> bool:
        """Check if counterfactual achieves target class.
        
        Args:
            counterfactual: Counterfactual instance.
            model: Trained model.
            target_class: Target class.
            
        Returns:
            True if counterfactual achieves target class.
        """
        prediction = model.predict([counterfactual])[0]
        return prediction == target_class
    
    @staticmethod
    def feasibility_score(
        counterfactual: np.ndarray,
        feature_ranges: Dict[str, Tuple[float, float]],
        feature_names: List[str]
    ) -> float:
        """Calculate feasibility score based on feature ranges.
        
        Args:
            counterfactual: Counterfactual instance.
            feature_ranges: Dictionary mapping feature names to (min, max) ranges.
            feature_names: List of feature names.
            
        Returns:
            Feasibility score (1.0 = fully feasible, 0.0 = infeasible).
        """
        feasible_features = 0
        total_features = len(counterfactual)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in feature_ranges:
                min_val, max_val = feature_ranges[feature_name]
                if min_val <= counterfactual[i] <= max_val:
                    feasible_features += 1
        
        return feasible_features / total_features
    
    @staticmethod
    def diversity_score(counterfactuals: List[np.ndarray]) -> float:
        """Calculate diversity score among multiple counterfactuals.
        
        Args:
            counterfactuals: List of counterfactual instances.
            
        Returns:
            Diversity score (higher is better).
        """
        if len(counterfactuals) < 2:
            return 0.0
        
        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                dist = np.linalg.norm(counterfactuals[i] - counterfactuals[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0


class ExplanationEvaluator:
    """Comprehensive evaluator for counterfactual explanations."""
    
    def __init__(self, feature_names: List[str], feature_ranges: Dict[str, Tuple[float, float]]):
        """Initialize evaluator.
        
        Args:
            feature_names: List of feature names.
            feature_ranges: Dictionary mapping feature names to (min, max) ranges.
        """
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
        self.metrics = CounterfactualMetrics()
    
    def evaluate_single(
        self, 
        original: np.ndarray, 
        counterfactual: np.ndarray, 
        model: Any, 
        target_class: int
    ) -> Dict[str, float]:
        """Evaluate a single counterfactual explanation.
        
        Args:
            original: Original instance.
            counterfactual: Counterfactual instance.
            model: Trained model.
            target_class: Target class.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        return {
            'proximity_euclidean': self.metrics.proximity_score(original, counterfactual, 'euclidean'),
            'proximity_manhattan': self.metrics.proximity_score(original, counterfactual, 'manhattan'),
            'proximity_cosine': self.metrics.proximity_score(original, counterfactual, 'cosine'),
            'sparsity': self.metrics.sparsity_score(original, counterfactual),
            'validity': float(self.metrics.validity_score(counterfactual, model, target_class)),
            'feasibility': self.metrics.feasibility_score(counterfactual, self.feature_ranges, self.feature_names)
        }
    
    def evaluate_multiple(
        self, 
        original: np.ndarray, 
        counterfactuals: List[np.ndarray], 
        model: Any, 
        target_class: int
    ) -> Dict[str, float]:
        """Evaluate multiple counterfactual explanations.
        
        Args:
            original: Original instance.
            counterfactuals: List of counterfactual instances.
            model: Trained model.
            target_class: Target class.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        single_metrics = [self.evaluate_single(original, cf, model, target_class) for cf in counterfactuals]
        
        # Aggregate metrics
        aggregated = {}
        for metric in single_metrics[0].keys():
            values = [m[metric] for m in single_metrics]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        # Add diversity score
        aggregated['diversity'] = self.metrics.diversity_score(counterfactuals)
        
        return aggregated
    
    def create_leaderboard(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create a leaderboard from evaluation results.
        
        Args:
            results: Dictionary mapping method names to evaluation results.
            
        Returns:
            DataFrame with leaderboard.
        """
        leaderboard_data = []
        
        for method_name, method_results in results.items():
            row = {'method': method_name}
            
            # Extract key metrics
            if 'proximity_euclidean_mean' in method_results:
                row['proximity'] = method_results['proximity_euclidean_mean']
                row['sparsity'] = method_results['sparsity_mean']
                row['validity'] = method_results['validity_mean']
                row['feasibility'] = method_results['feasibility_mean']
                row['diversity'] = method_results.get('diversity', 0.0)
            else:
                # Single counterfactual results
                row['proximity'] = method_results.get('proximity_euclidean', 0.0)
                row['sparsity'] = method_results.get('sparsity', 0.0)
                row['validity'] = method_results.get('validity', 0.0)
                row['feasibility'] = method_results.get('feasibility', 0.0)
                row['diversity'] = 0.0
            
            leaderboard_data.append(row)
        
        df = pd.DataFrame(leaderboard_data)
        
        # Sort by composite score (lower proximity and sparsity, higher validity and feasibility)
        df['composite_score'] = (
            -df['proximity'] - df['sparsity'] + 
            df['validity'] * 10 + df['feasibility'] * 5
        )
        df = df.sort_values('composite_score', ascending=False)
        
        return df
