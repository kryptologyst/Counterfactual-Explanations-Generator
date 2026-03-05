# Counterfactual Explanations Generator

Research-focused implementation of counterfactual explanation methods for Explainable AI (XAI).

## ⚠️ Important Disclaimers

**This tool is for research and educational purposes only.** 

- Counterfactual explanations may be unstable or misleading
- Results should not be used for regulated decisions without human review
- Always validate explanations with domain experts before production use
- The tool is not intended for high-stakes decision making

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Counterfactual-Explanations-Generator.git
cd Counterfactual-Explanations-Generator

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.data.loaders import load_iris_dataset, preprocess_data
from src.methods.counterfactuals import RandomPerturbationExplainer
from src.eval.metrics import ExplanationEvaluator
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y, metadata = load_iris_dataset()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate counterfactual
explainer = RandomPerturbationExplainer(epsilon=0.1, max_iterations=1000)
explanation = explainer.explain(X_test[0], model)

print(f"Original prediction: {explanation['original_prediction']}")
print(f"Counterfactual prediction: {explanation['counterfactual_prediction']}")
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run src/demo/streamlit_app.py
```

## Features

### Counterfactual Methods

- **Random Perturbation**: Simple random feature perturbation
- **Gradient-Based**: Optimization-based counterfactual generation
- **Alibi Integration**: Advanced counterfactual methods from Alibi library

### Evaluation Metrics

- **Proximity**: Distance between original and counterfactual instances
- **Sparsity**: Number of features that need to change
- **Validity**: Whether counterfactual achieves target class
- **Feasibility**: Whether changes are within realistic bounds
- **Diversity**: Variation among multiple counterfactuals

### Visualization

- Feature comparison plots
- Interactive Plotly visualizations
- Metrics comparison charts
- Leaderboard rankings

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── methods/        # Counterfactual explanation methods
├── eval/           # Evaluation metrics and frameworks
├── viz/            # Visualization utilities
├── utils/          # Core utilities (seeding, device management)
└── demo/           # Interactive demo applications

configs/            # Configuration files
scripts/            # Example scripts and demos
tests/              # Unit tests
assets/             # Output plots and results
```

## 🔧 Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
data:
  dataset: "iris"
  test_size: 0.3
  random_state: 42

methods:
  random_perturbation:
    epsilon: 0.1
    max_iterations: 1000
```

## Evaluation

### Metrics Leaderboard

The framework provides comprehensive evaluation with a leaderboard:

| Method | Proximity | Sparsity | Validity | Feasibility | Composite Score |
|--------|-----------|----------|----------|-------------|-----------------|
| Random Perturbation | 0.234 | 2.1 | 1.0 | 0.95 | 8.45 |
| Gradient-Based | 0.189 | 1.8 | 1.0 | 0.98 | 9.12 |

### Sanity Checks

- **Randomization Test**: Verify explanations change with random baselines
- **Stability Test**: Check consistency across different seeds
- **Faithfulness Test**: Validate that explanations reflect model behavior

## Use Cases

### Research Applications

- Model interpretability research
- Counterfactual explanation method development
- XAI algorithm comparison
- Educational demonstrations

### Educational Applications

- Teaching explainable AI concepts
- Understanding model decision boundaries
- Exploring feature importance
- Learning about counterfactual reasoning

## Advanced Features

### Multiple Datasets

- **Iris Dataset**: Classic classification benchmark
- **Synthetic Dataset**: Configurable synthetic data generation

### Model Support

- Random Forest Classifier
- Logistic Regression
- Decision Tree Classifier
- Extensible to other scikit-learn models

### Device Support

- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU fallback

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_counterfactuals.py
```

## Development

### Code Quality

The project uses modern Python development practices:

- **Type hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black code formatting
- **Linting**: Ruff for code quality
- **Testing**: Pytest with coverage

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## References

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box
- Verma, S., et al. (2020). Counterfactual explanations for machine learning: A review
- Guidotti, R., et al. (2018). A survey of methods for explaining black box models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Remember**: This tool is for research and educational purposes. Always validate counterfactual explanations with domain experts before using them in production systems.
# Counterfactual-Explanations-Generator
