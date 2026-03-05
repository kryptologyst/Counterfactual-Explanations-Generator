"""Data loading and preprocessing utilities."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class DatasetMetadata:
    """Metadata container for datasets."""
    
    def __init__(
        self,
        feature_names: List[str],
        target_names: List[str],
        feature_types: List[str],
        feature_ranges: Dict[str, Tuple[float, float]],
        sensitive_attributes: Optional[List[str]] = None,
        monotonic_features: Optional[List[str]] = None,
    ):
        """Initialize dataset metadata.
        
        Args:
            feature_names: List of feature names.
            target_names: List of target class names.
            feature_types: List of feature types ('continuous', 'categorical', 'binary').
            feature_ranges: Dictionary mapping feature names to (min, max) ranges.
            sensitive_attributes: List of sensitive attribute names.
            monotonic_features: List of features with monotonic constraints.
        """
        self.feature_names = feature_names
        self.target_names = target_names
        self.feature_types = feature_types
        self.feature_ranges = feature_ranges
        self.sensitive_attributes = sensitive_attributes or []
        self.monotonic_features = monotonic_features or []


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Load and prepare Iris dataset.
    
    Returns:
        Tuple of (X, y, metadata).
    """
    data = load_iris()
    X = data.data
    y = data.target
    
    metadata = DatasetMetadata(
        feature_names=data.feature_names,
        target_names=data.target_names,
        feature_types=['continuous'] * len(data.feature_names),
        feature_ranges={
            name: (X[:, i].min(), X[:, i].max()) 
            for i, name in enumerate(data.feature_names)
        }
    )
    
    return X, y, metadata


def load_synthetic_dataset(
    n_samples: int = 1000,
    n_features: int = 4,
    n_classes: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, DatasetMetadata]:
    """Generate synthetic dataset for testing.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        n_classes: Number of classes.
        random_state: Random seed.
        
    Returns:
        Tuple of (X, y, metadata).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=random_state
    )
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    target_names = [f"class_{i}" for i in range(n_classes)]
    
    metadata = DatasetMetadata(
        feature_names=feature_names,
        target_names=target_names,
        feature_types=['continuous'] * n_features,
        feature_ranges={
            name: (X[:, i].min(), X[:, i].max()) 
            for i, name in enumerate(feature_names)
        }
    )
    
    return X, y, metadata


def preprocess_data(
    X: np.ndarray,
    y: np.ndarray,
    scaler_type: str = "minmax",
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Preprocess data with scaling and train/test split.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        scaler_type: Type of scaler ('minmax' or 'standard').
        test_size: Proportion of data for testing.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
