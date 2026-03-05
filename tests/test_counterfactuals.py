"""Tests for counterfactual explanations."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.data.loaders import load_iris_dataset, preprocess_data
from src.methods.counterfactuals import RandomPerturbationExplainer, GradientBasedExplainer
from src.eval.metrics import CounterfactualMetrics, ExplanationEvaluator
from src.utils.core import set_seed


class TestCounterfactualMetrics:
    """Test counterfactual evaluation metrics."""
    
    def test_proximity_score(self):
        """Test proximity score calculation."""
        original = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.5, 2.5, 3.5])
        
        euclidean = CounterfactualMetrics.proximity_score(original, counterfactual, "euclidean")
        manhattan = CounterfactualMetrics.proximity_score(original, counterfactual, "manhattan")
        
        assert euclidean > 0
        assert manhattan > 0
        assert manhattan > euclidean  # Manhattan should be larger than Euclidean
    
    def test_sparsity_score(self):
        """Test sparsity score calculation."""
        original = np.array([1.0, 2.0, 3.0, 4.0])
        counterfactual = np.array([1.0, 2.5, 3.0, 4.5])
        
        sparsity = CounterfactualMetrics.sparsity_score(original, counterfactual)
        
        assert sparsity == 2  # Two features changed
    
    def test_validity_score(self):
        """Test validity score calculation."""
        # Create a simple model for testing
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test with correct prediction
        instance = X[0]
        prediction = model.predict([instance])[0]
        
        validity = CounterfactualMetrics.validity_score(instance, model, prediction)
        assert validity is True
        
        # Test with incorrect prediction
        wrong_class = 1 - prediction
        validity_wrong = CounterfactualMetrics.validity_score(instance, model, wrong_class)
        assert validity_wrong is False


class TestRandomPerturbationExplainer:
    """Test random perturbation explainer."""
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        explainer = RandomPerturbationExplainer(epsilon=0.1, max_iterations=100)
        
        assert explainer.epsilon == 0.1
        assert explainer.max_iterations == 100
        assert explainer.random_state == 42
    
    def test_explain_method(self):
        """Test explain method."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Create test data and model
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Initialize explainer
        explainer = RandomPerturbationExplainer(epsilon=0.1, max_iterations=100)
        
        # Generate explanation
        instance = X_test[0]
        explanation = explainer.explain(instance, model)
        
        # Check explanation structure
        assert 'counterfactual' in explanation
        assert 'original_prediction' in explanation
        assert 'counterfactual_prediction' in explanation
        assert 'target_class' in explanation
        assert 'iterations' in explanation
        assert 'feature_changes' in explanation
        assert 'success' in explanation
        
        # Check data types
        assert isinstance(explanation['counterfactual'], np.ndarray)
        assert isinstance(explanation['original_prediction'], (int, np.integer))
        assert isinstance(explanation['counterfactual_prediction'], (int, np.integer))
        assert isinstance(explanation['iterations'], int)
        assert isinstance(explanation['success'], bool)


class TestExplanationEvaluator:
    """Test explanation evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        feature_names = ['feature1', 'feature2', 'feature3']
        feature_ranges = {
            'feature1': (0.0, 1.0),
            'feature2': (0.0, 1.0),
            'feature3': (0.0, 1.0)
        }
        
        evaluator = ExplanationEvaluator(feature_names, feature_ranges)
        
        assert evaluator.feature_names == feature_names
        assert evaluator.feature_ranges == feature_ranges
    
    def test_evaluate_single(self):
        """Test single explanation evaluation."""
        # Set seed for reproducibility
        set_seed(42)
        
        # Create test data and model
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Initialize evaluator
        feature_names = [f'feature_{i}' for i in range(4)]
        feature_ranges = {name: (0.0, 1.0) for name in feature_names}
        evaluator = ExplanationEvaluator(feature_names, feature_ranges)
        
        # Generate explanation
        explainer = RandomPerturbationExplainer(epsilon=0.1, max_iterations=100)
        instance = X_test[0]
        explanation = explainer.explain(instance, model)
        
        # Evaluate
        metrics = evaluator.evaluate_single(
            instance,
            explanation['counterfactual'],
            model,
            explanation['target_class']
        )
        
        # Check metrics structure
        expected_metrics = [
            'proximity_euclidean',
            'proximity_manhattan',
            'proximity_cosine',
            'sparsity',
            'validity',
            'feasibility'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_create_leaderboard(self):
        """Test leaderboard creation."""
        feature_names = ['feature1', 'feature2']
        feature_ranges = {'feature1': (0.0, 1.0), 'feature2': (0.0, 1.0)}
        
        evaluator = ExplanationEvaluator(feature_names, feature_ranges)
        
        # Mock results
        results = {
            'method1': {
                'proximity_euclidean': 0.5,
                'sparsity': 2.0,
                'validity': 1.0,
                'feasibility': 0.8
            },
            'method2': {
                'proximity_euclidean': 0.3,
                'sparsity': 1.5,
                'validity': 1.0,
                'feasibility': 0.9
            }
        }
        
        leaderboard = evaluator.create_leaderboard(results)
        
        assert len(leaderboard) == 2
        assert 'method' in leaderboard.columns
        assert 'composite_score' in leaderboard.columns
        assert leaderboard.iloc[0]['composite_score'] >= leaderboard.iloc[1]['composite_score']


class TestDataLoaders:
    """Test data loading utilities."""
    
    def test_load_iris_dataset(self):
        """Test Iris dataset loading."""
        X, y, metadata = load_iris_dataset()
        
        assert X.shape[1] == 4  # 4 features
        assert len(np.unique(y)) == 3  # 3 classes
        assert len(metadata.feature_names) == 4
        assert len(metadata.target_names) == 3
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert y_train.shape[0] + y_test.shape[0] == y.shape[0]
        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]


if __name__ == "__main__":
    pytest.main([__file__])
