"""Streamlit demo application for counterfactual explanations."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loaders import load_iris_dataset, load_synthetic_dataset, preprocess_data
from src.methods.counterfactuals import (
    RandomPerturbationExplainer, 
    GradientBasedExplainer,
    AlibiCounterfactualExplainer
)
from src.eval.metrics import ExplanationEvaluator
from src.viz.visualizer import CounterfactualVisualizer
from src.utils.core import set_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Counterfactual Explanations Generator",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header with disclaimer
    st.title("🔍 Counterfactual Explanations Generator")
    st.markdown("""
    **DISCLAIMER**: This tool is for research and educational purposes only. 
    Counterfactual explanations may be unstable or misleading and should not be used 
    for regulated decisions without human review.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox(
        "Select Dataset",
        ["Iris Dataset", "Synthetic Dataset"]
    )
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Logistic Regression", "Decision Tree"]
    )
    
    # Method selection
    method_option = st.sidebar.selectbox(
        "Select Counterfactual Method",
        ["Random Perturbation", "Gradient-Based", "Alibi"]
    )
    
    # Parameters
    st.sidebar.subheader("Parameters")
    epsilon = st.sidebar.slider("Epsilon (perturbation magnitude)", 0.01, 0.5, 0.1)
    max_iterations = st.sidebar.slider("Max Iterations", 100, 2000, 1000)
    random_seed = st.sidebar.number_input("Random Seed", 42, 9999, 42)
    
    # Load data and train model
    if st.sidebar.button("Load Data & Train Model"):
        with st.spinner("Loading data and training model..."):
            # Set seed for reproducibility
            set_seed(random_seed)
            
            # Load dataset
            if dataset_option == "Iris Dataset":
                X, y, metadata = load_iris_dataset()
            else:
                X, y, metadata = load_synthetic_dataset(random_state=random_seed)
            
            # Preprocess data
            X_train, X_test, y_train, y_test, scaler = preprocess_data(
                X, y, random_state=random_seed
            )
            
            # Train model
            if model_option == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=random_seed)
            elif model_option == "Logistic Regression":
                model = LogisticRegression(random_state=random_seed, max_iter=1000)
            else:
                model = DecisionTreeClassifier(random_state=random_seed)
            
            model.fit(X_train, y_train)
            
            # Store in session state
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.session_state.metadata = metadata
            st.session_state.scaler = scaler
            
            st.success("Model trained successfully!")
    
    # Main content
    if 'model' in st.session_state:
        st.header("Counterfactual Explanation")
        
        # Instance selection
        st.subheader("Select Instance to Explain")
        
        # Show test instances
        test_df = pd.DataFrame(
            st.session_state.X_test,
            columns=st.session_state.metadata.feature_names
        )
        test_df['prediction'] = st.session_state.model.predict(st.session_state.X_test)
        test_df['target'] = st.session_state.y_test
        
        # Instance selector
        instance_idx = st.selectbox(
            "Select test instance",
            range(len(st.session_state.X_test)),
            format_func=lambda x: f"Instance {x}: Predicted {st.session_state.metadata.target_names[test_df.iloc[x]['prediction']]}"
        )
        
        selected_instance = st.session_state.X_test[instance_idx]
        original_pred = st.session_state.model.predict([selected_instance])[0]
        
        # Display original instance
        st.subheader("Original Instance")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature values
            feature_df = pd.DataFrame({
                'Feature': st.session_state.metadata.feature_names,
                'Value': selected_instance
            })
            st.dataframe(feature_df, use_container_width=True)
        
        with col2:
            st.metric("Predicted Class", st.session_state.metadata.target_names[original_pred])
            st.metric("True Class", st.session_state.metadata.target_names[st.session_state.y_test[instance_idx]])
        
        # Generate counterfactual
        if st.button("Generate Counterfactual Explanation"):
            with st.spinner("Generating counterfactual explanation..."):
                # Initialize explainer
                if method_option == "Random Perturbation":
                    explainer = RandomPerturbationExplainer(
                        epsilon=epsilon,
                        max_iterations=max_iterations,
                        random_state=random_seed
                    )
                elif method_option == "Gradient-Based":
                    explainer = GradientBasedExplainer(
                        max_iterations=max_iterations,
                        random_state=random_seed
                    )
                else:  # Alibi
                    explainer = AlibiCounterfactualExplainer(
                        max_iterations=max_iterations,
                        random_state=random_seed
                    )
                
                # Generate explanation
                explanation = explainer.explain(selected_instance, st.session_state.model)
                
                # Store in session state
                st.session_state.explanation = explanation
                
                st.success("Counterfactual explanation generated!")
        
        # Display results
        if 'explanation' in st.session_state:
            st.subheader("Counterfactual Explanation Results")
            
            explanation = st.session_state.explanation
            counterfactual = explanation['counterfactual']
            cf_pred = explanation['counterfactual_prediction']
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Original Prediction", 
                    st.session_state.metadata.target_names[explanation['original_prediction']]
                )
            
            with col2:
                st.metric(
                    "Counterfactual Prediction", 
                    st.session_state.metadata.target_names[cf_pred]
                )
            
            with col3:
                st.metric("Success", "✓" if explanation['success'] else "✗")
            
            with col4:
                st.metric("Iterations", explanation['iterations'])
            
            # Feature comparison
            st.subheader("Feature Comparison")
            
            # Create comparison plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Original", "Counterfactual"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Original
            fig.add_trace(
                go.Bar(
                    x=st.session_state.metadata.feature_names,
                    y=selected_instance,
                    name="Original",
                    marker_color='skyblue',
                    text=[f"{val:.3f}" for val in selected_instance],
                    textposition='auto',
                ),
                row=1, col=1
            )
            
            # Counterfactual
            fig.add_trace(
                go.Bar(
                    x=st.session_state.metadata.feature_names,
                    y=counterfactual,
                    name="Counterfactual",
                    marker_color='lightcoral',
                    text=[f"{val:.3f}" for val in counterfactual],
                    textposition='auto',
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="Feature Value Comparison",
                showlegend=False,
                height=500,
                width=1000
            )
            
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(title_text="Feature Value")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature changes
            st.subheader("Feature Changes")
            changes = counterfactual - selected_instance
            
            changes_df = pd.DataFrame({
                'Feature': st.session_state.metadata.feature_names,
                'Change': changes,
                'Absolute Change': np.abs(changes)
            }).sort_values('Absolute Change', ascending=False)
            
            st.dataframe(changes_df, use_container_width=True)
            
            # Evaluation metrics
            st.subheader("Evaluation Metrics")
            
            evaluator = ExplanationEvaluator(
                st.session_state.metadata.feature_names,
                st.session_state.metadata.feature_ranges
            )
            
            metrics = evaluator.evaluate_single(
                selected_instance,
                counterfactual,
                st.session_state.model,
                explanation['target_class']
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Proximity (Euclidean)", f"{metrics['proximity_euclidean']:.3f}")
            
            with col2:
                st.metric("Sparsity", f"{metrics['sparsity']:.1f}")
            
            with col3:
                st.metric("Validity", f"{metrics['validity']:.1f}")
            
            with col4:
                st.metric("Feasibility", f"{metrics['feasibility']:.3f}")
            
            # Additional information
            st.subheader("Additional Information")
            
            if 'proximity' in explanation:
                st.write(f"**Proximity Score**: {explanation['proximity']:.3f}")
            
            if 'sparsity' in explanation:
                st.write(f"**Sparsity Score**: {explanation['sparsity']:.1f}")
            
            st.write(f"**Target Class**: {st.session_state.metadata.target_names[explanation['target_class']]}")
            
            # Download results
            st.subheader("Download Results")
            
            results_data = {
                'original': selected_instance.tolist(),
                'counterfactual': counterfactual.tolist(),
                'feature_names': st.session_state.metadata.feature_names,
                'metrics': metrics,
                'explanation': explanation
            }
            
            import json
            results_json = json.dumps(results_data, indent=2)
            
            st.download_button(
                label="Download Results (JSON)",
                data=results_json,
                file_name=f"counterfactual_results_{instance_idx}.json",
                mime="application/json"
            )
    
    else:
        st.info("Please load data and train a model using the sidebar controls.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: This is a research tool for educational purposes. 
    Always validate counterfactual explanations with domain experts before 
    using them in production systems.
    """)


if __name__ == "__main__":
    main()
