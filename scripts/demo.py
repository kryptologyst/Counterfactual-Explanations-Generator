"""Main script demonstrating modernized counterfactual explanations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loaders import load_iris_dataset, preprocess_data
from src.methods.counterfactuals import (
    RandomPerturbationExplainer, 
    GradientBasedExplainer
)
from src.eval.metrics import ExplanationEvaluator
from src.viz.visualizer import CounterfactualVisualizer
from src.utils.core import set_seed, get_device_name
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def main():
    """Main demonstration script."""
    print("🔍 Modern Counterfactual Explanations Generator")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    print(f"Device: {get_device_name()}")
    print(f"Random seed set to 42 for reproducibility")
    print()
    
    # Load and preprocess data
    print("📊 Loading Iris dataset...")
    X, y, metadata = load_iris_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {metadata.feature_names}")
    print(f"Classes: {metadata.target_names}")
    print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
    print()
    
    # Train multiple models
    print("🤖 Training models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"{name}: Train Acc = {train_acc:.3f}, Test Acc = {test_acc:.3f}")
    print()
    
    # Initialize explainers
    print("🔧 Initializing counterfactual explainers...")
    explainers = {
        'Random Perturbation': RandomPerturbationExplainer(epsilon=0.1, max_iterations=1000),
        'Gradient-Based': GradientBasedExplainer(max_iterations=1000)
    }
    
    # Initialize evaluator and visualizer
    evaluator = ExplanationEvaluator(metadata.feature_names, metadata.feature_ranges)
    visualizer = CounterfactualVisualizer(metadata.feature_names, metadata.target_names)
    
    # Evaluate on a sample instance
    sample_idx = 0
    sample_instance = X_test[sample_idx]
    original_pred = models['Random Forest'].predict([sample_instance])[0]
    
    print(f"Sample instance {sample_idx}:")
    print(f"Features: {dict(zip(metadata.feature_names, sample_instance))}")
    print(f"True class: {metadata.target_names[y_test[sample_idx]]}")
    print(f"Predicted class: {metadata.target_names[original_pred]}")
    print()
    
    # Generate counterfactuals for each method
    results = {}
    
    for method_name, explainer in explainers.items():
        print(f"🔍 Generating counterfactual using {method_name}...")
        
        explanation = explainer.explain(sample_instance, models['Random Forest'])
        counterfactual = explanation['counterfactual']
        
        # Evaluate the counterfactual
        metrics = evaluator.evaluate_single(
            sample_instance,
            counterfactual,
            models['Random Forest'],
            explanation['target_class']
        )
        
        results[method_name] = {
            'explanation': explanation,
            'metrics': metrics
        }
        
        print(f"  Success: {explanation['success']}")
        print(f"  Iterations: {explanation['iterations']}")
        print(f"  Target class: {metadata.target_names[explanation['target_class']]}")
        print(f"  Proximity: {metrics['proximity_euclidean']:.3f}")
        print(f"  Sparsity: {metrics['sparsity']:.1f}")
        print(f"  Validity: {metrics['validity']:.1f}")
        print(f"  Feasibility: {metrics['feasibility']:.3f}")
        print()
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # Feature comparison plots
    for method_name, result in results.items():
        explanation = result['explanation']
        counterfactual = explanation['counterfactual']
        
        fig = visualizer.plot_feature_comparison(
            sample_instance,
            counterfactual,
            title=f"{method_name} - Original vs Counterfactual"
        )
        
        plt.savefig(f"assets/{method_name.lower().replace(' ', '_')}_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature changes plot
        fig = visualizer.plot_feature_changes(
            sample_instance,
            counterfactual,
            title=f"{method_name} - Feature Changes"
        )
        
        plt.savefig(f"assets/{method_name.lower().replace(' ', '_')}_changes.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # Metrics comparison
    metrics_data = {name: result['metrics'] for name, result in results.items()}
    fig = visualizer.plot_metrics_comparison(metrics_data)
    plt.savefig("assets/metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create leaderboard
    leaderboard_df = evaluator.create_leaderboard(metrics_data)
    print("🏆 Leaderboard:")
    print(leaderboard_df.to_string(index=False))
    print()
    
    # Save leaderboard
    leaderboard_df.to_csv("assets/leaderboard.csv", index=False)
    
    # Plot leaderboard
    fig = visualizer.plot_leaderboard(leaderboard_df)
    plt.savefig("assets/leaderboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Analysis complete!")
    print("📁 Results saved to assets/ directory")
    print()
    
    # Display summary statistics
    print("📈 Summary Statistics:")
    print(f"Best method: {leaderboard_df.iloc[0]['method']}")
    print(f"Best composite score: {leaderboard_df.iloc[0]['composite_score']:.3f}")
    
    # Feature importance analysis
    print("\n🔍 Feature Importance Analysis:")
    rf_model = models['Random Forest']
    feature_importance = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': metadata.feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print(importance_df.to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv("assets/feature_importance.csv", index=False)
    
    print("\n🎯 Key Insights:")
    print("1. Counterfactual explanations show how to change predictions")
    print("2. Proximity measures how close the counterfactual is to the original")
    print("3. Sparsity measures how many features need to change")
    print("4. Validity ensures the counterfactual achieves the target class")
    print("5. Feasibility checks if changes are within realistic bounds")
    
    print("\n⚠️  Important Disclaimers:")
    print("- These explanations are for research and educational purposes only")
    print("- Counterfactuals may be unstable or misleading")
    print("- Always validate with domain experts before production use")
    print("- Do not use for regulated decisions without human review")


if __name__ == "__main__":
    main()
