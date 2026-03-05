"""Advanced counterfactual explanation methods."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
try:
    from alibi.explainers import CounterfactualRLTabular
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False
    CounterfactualRLTabular = None


class BaseCounterfactualExplainer(ABC):
    """Abstract base class for counterfactual explainers."""
    
    @abstractmethod
    def explain(
        self, 
        instance: np.ndarray, 
        model: Any,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation.
        
        Args:
            instance: Input instance to explain.
            model: Trained model.
            target_class: Target class for counterfactual (if None, uses opposite class).
            
        Returns:
            Dictionary containing counterfactual explanation.
        """
        pass


class RandomPerturbationExplainer(BaseCounterfactualExplainer):
    """Simple random perturbation counterfactual explainer."""
    
    def __init__(
        self,
        epsilon: float = 0.1,
        max_iterations: int = 1000,
        random_state: int = 42
    ):
        """Initialize random perturbation explainer.
        
        Args:
            epsilon: Perturbation magnitude.
            max_iterations: Maximum number of iterations.
            random_state: Random seed.
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.random_state = random_state
        np.random.seed(random_state)
    
    def explain(
        self, 
        instance: np.ndarray, 
        model: Any,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate counterfactual using random perturbation.
        
        Args:
            instance: Input instance to explain.
            model: Trained model.
            target_class: Target class for counterfactual.
            
        Returns:
            Dictionary containing counterfactual explanation.
        """
        original_pred = model.predict([instance])[0]
        
        if target_class is None:
            # Find a different class
            if hasattr(model, 'classes_'):
                all_classes = model.classes_
            else:
                # Fallback: try to predict on a few samples to get classes
                all_classes = np.unique(model.predict(instance.reshape(1, -1)))
            
            if len(all_classes) > 1:
                target_class = all_classes[all_classes != original_pred][0]
            else:
                # If only one class, cycle through possible classes
                target_class = (original_pred + 1) % 3  # Assuming max 3 classes
        
        counterfactual = np.copy(instance)
        iterations = 0
        
        while iterations < self.max_iterations:
            if model.predict([counterfactual])[0] == target_class:
                break
                
            # Randomly perturb features
            feature_idx = np.random.randint(0, len(instance))
            perturbation = np.random.uniform(-self.epsilon, self.epsilon)
            counterfactual[feature_idx] += perturbation
            
            # Ensure bounds
            counterfactual = np.clip(counterfactual, 0, 1)
            iterations += 1
        
        return {
            'counterfactual': counterfactual,
            'original_prediction': original_pred,
            'counterfactual_prediction': model.predict([counterfactual])[0],
            'target_class': target_class,
            'iterations': iterations,
            'feature_changes': counterfactual - instance,
            'success': iterations < self.max_iterations
        }


class GradientBasedExplainer(BaseCounterfactualExplainer):
    """Gradient-based counterfactual explainer."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        lambda_proximity: float = 1.0,
        lambda_sparsity: float = 0.1,
        random_state: int = 42
    ):
        """Initialize gradient-based explainer.
        
        Args:
            learning_rate: Learning rate for optimization.
            max_iterations: Maximum number of iterations.
            lambda_proximity: Weight for proximity loss.
            lambda_sparsity: Weight for sparsity loss.
            random_state: Random seed.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lambda_proximity = lambda_proximity
        self.lambda_sparsity = lambda_sparsity
        self.random_state = random_state
        np.random.seed(random_state)
    
    def explain(
        self, 
        instance: np.ndarray, 
        model: Any,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate counterfactual using gradient-based optimization.
        
        Args:
            instance: Input instance to explain.
            model: Trained model.
            target_class: Target class for counterfactual.
            
        Returns:
            Dictionary containing counterfactual explanation.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch library is not available. Please install it with: pip install torch")
        
        original_pred = model.predict([instance])[0]
        
        if target_class is None:
            # Find a different class
            if hasattr(model, 'classes_'):
                all_classes = model.classes_
            else:
                # Fallback: try to predict on a few samples to get classes
                all_classes = np.unique(model.predict(instance.reshape(1, -1)))
            
            if len(all_classes) > 1:
                target_class = all_classes[all_classes != original_pred][0]
            else:
                # If only one class, cycle through possible classes
                target_class = (original_pred + 1) % 3  # Assuming max 3 classes
        
        # Convert to PyTorch tensors
        instance_tensor = torch.tensor(instance, dtype=torch.float32, requires_grad=True)
        target_tensor = torch.tensor(target_class, dtype=torch.long)
        
        optimizer = torch.optim.Adam([instance_tensor], lr=self.learning_rate)
        
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # Prediction loss (simplified - would need proper model wrapper)
            pred_loss = torch.nn.functional.mse_loss(
                instance_tensor, 
                torch.tensor(target_class, dtype=torch.float32)
            )
            
            # Proximity loss
            proximity_loss = torch.nn.functional.mse_loss(instance_tensor, torch.tensor(instance))
            
            # Sparsity loss
            sparsity_loss = torch.sum(torch.abs(instance_tensor - torch.tensor(instance)))
            
            # Total loss
            total_loss = pred_loss + self.lambda_proximity * proximity_loss + self.lambda_sparsity * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Check if target class is reached
            if iteration % 100 == 0:
                current_pred = model.predict([instance_tensor.detach().numpy()])[0]
                if current_pred == target_class:
                    break
        
        counterfactual = instance_tensor.detach().numpy()
        
        return {
            'counterfactual': counterfactual,
            'original_prediction': original_pred,
            'counterfactual_prediction': model.predict([counterfactual])[0],
            'target_class': target_class,
            'iterations': iteration + 1,
            'feature_changes': counterfactual - instance,
            'success': model.predict([counterfactual])[0] == target_class
        }


class AlibiCounterfactualExplainer(BaseCounterfactualExplainer):
    """Wrapper for Alibi counterfactual explainer."""
    
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        max_iterations: int = 1000,
        random_state: int = 42
    ):
        """Initialize Alibi counterfactual explainer.
        
        Args:
            feature_range: Range of feature values.
            max_iterations: Maximum number of iterations.
            random_state: Random seed.
        """
        self.feature_range = feature_range
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.explainer = None
    
    def explain(
        self, 
        instance: np.ndarray, 
        model: Any,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate counterfactual using Alibi.
        
        Args:
            instance: Input instance to explain.
            model: Trained model.
            target_class: Target class for counterfactual.
            
        Returns:
            Dictionary containing counterfactual explanation.
        """
        if not ALIBI_AVAILABLE:
            raise ImportError("Alibi library is not available. Please install it with: pip install alibi")
        
        if self.explainer is None:
            # Initialize explainer
            self.explainer = CounterfactualRLTabular(
                model,
                feature_range=self.feature_range,
                max_iterations=self.max_iterations,
                random_state=self.random_state
            )
        
        original_pred = model.predict([instance])[0]
        
        if target_class is None:
            # Find a different class
            if hasattr(model, 'classes_'):
                all_classes = model.classes_
            else:
                # Fallback: try to predict on a few samples to get classes
                all_classes = np.unique(model.predict(instance.reshape(1, -1)))
            
            if len(all_classes) > 1:
                target_class = all_classes[all_classes != original_pred][0]
            else:
                # If only one class, cycle through possible classes
                target_class = (original_pred + 1) % 3  # Assuming max 3 classes
        
        # Generate counterfactual
        explanation = self.explainer.explain(
            instance.reshape(1, -1),
            target_class=target_class
        )
        
        counterfactual = explanation.cf['X'][0]
        
        return {
            'counterfactual': counterfactual,
            'original_prediction': original_pred,
            'counterfactual_prediction': model.predict([counterfactual])[0],
            'target_class': target_class,
            'iterations': explanation.cf.get('iterations', 0),
            'feature_changes': counterfactual - instance,
            'success': explanation.cf.get('success', False),
            'proximity': explanation.cf.get('proximity', 0.0),
            'sparsity': explanation.cf.get('sparsity', 0.0)
        }
