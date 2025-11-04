# Titanic Survival Prediction - MLOps Ready

[![codecov](https://codecov.io/gh/Thami279/-titanic-random-forest/branch/main/graph/badge.svg)](https://codecov.io/gh/Thami279/-titanic-random-forest)

Production-ready machine learning package for predicting Titanic passenger survival with comprehensive testing, experiment tracking, and deployment capabilities.

## Features

- ✅ **Feature Engineering**: Domain-specific feature extraction (titles, family metrics, ticket/cabin signals)
- ✅ **Preprocessing Pipeline**: Reusable, tested data preprocessing utilities
- ✅ **Model Training**: Random Forest, Bagging, and Boosting with hyperparameter tuning
- ✅ **Experiment Tracking**: MLflow integration for model versioning and comparison
- ✅ **Data Validation**: Input validation with range checks and categorical constraints
- ✅ **Unit & Integration Tests**: Comprehensive pytest test suite (82% coverage)
- ✅ **Reproducibility**: Locked dependency versions and dataset fingerprinting

## Project Structure

```
.
├── titanic_ml/           # Main package
│   ├── models/          # Model training and tracking
│   └── utils/           # Preprocessing and validation
├── tests/               # Test suite
├── random_forest_titantic.ipynb  # Main analysis notebook
├── requirements.txt     # Locked dependencies
├── setup.py             # Package setup
└── DEPLOYMENT.md        # Deployment guide

```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Command Line Training

Train all ensembles with data validation, performance budget enforcement, and
experiment tracking:

```bash
python -m titanic_ml train --data-path titanic.csv --output reports/latest.json
```

Add `--fast-mode` during experimentation to shrink the hyperparameter search space
for quicker feedback, or `--use-mlflow` to push runs to a configured MLflow server.

### Running the Notebook

```bash
jupyter notebook random_forest_titantic.ipynb
```

### Using the Package Programmatically

```python
from titanic_ml.utils.preprocessing import engineer_features, build_preprocessor
from titanic_ml.models.tracking import ExperimentTracker
from titanic_ml.utils.validation import validate_data

import pandas as pd

# Load data
df = pd.read_csv('titanic.csv')

# Engineer features
engineered = engineer_features(df)

# Validate data
validate_data(engineered, required_columns=['Pclass', 'Sex', 'Age'])

# Build preprocessor
preprocessor = build_preprocessor(
    numeric_features=['Age', 'Fare'],
    categorical_features=['Sex', 'Pclass']
)
```

### Serving Predictions (FastAPI)

```bash
uvicorn titanic_ml.api:app --host 0.0.0.0 --port 5000
```

The `/health` endpoint reports service status, `/predict` accepts raw passenger
records, and `/report` returns the latest training summary. `docker-compose up`
starts the API together with an MLflow tracking server.

## Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=titanic_ml --cov-report=html

# View coverage report
open htmlcov/index.html
```

## How to Reproduce

1. Clone the repository and `cd` into the project directory.
2. Create and activate a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
3. Install dependencies and the package in editable mode (`pip install -r requirements.txt && pip install -e .`).
4. Verify the environment with `pytest --cov=titanic_ml`.
5. Re-run training to match the reported metrics: `python -m titanic_ml train --fast-mode --data-path titanic.csv`.
6. Optionally launch the notebook (`jupyter notebook random_forest_titantic.ipynb`) to inspect intermediate outputs.

## Experiment Tracking

The package supports MLflow for experiment tracking:

```python
from titanic_ml.models.tracking import ExperimentTracker

# Initialize tracker (auto-detects MLflow)
tracker = ExperimentTracker(use_mlflow=True, experiment_name="titanic_survival")

# Log experiment
tracker.log(
    name="RandomForest_v1",
    estimator=model,
    y_true=y_test,
    y_pred=predictions,
    y_proba=probabilities,
    params=best_params,
    notes="Baseline model"
)

# View results
results_df = tracker.to_dataframe()
```

To start MLflow UI:
```bash
mlflow ui
# Visit http://localhost:5000
```

## Model Performance

Best model from recent training:
- **Model**: Bagging (DecisionTree)
- **Accuracy**: 79.89%
- **ROC-AUC**: 0.8513
- **Parameters**: n_estimators=50, max_depth=5

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:
- Batch scoring
- REST API serving
- Docker containerization
- Monitoring and alerting

## Requirements

- Python 3.9+
- See `requirements.txt` for exact dependency versions

## Testing Status

✅ **11 tests passing** with 82% code coverage

Test categories:
- Unit tests for preprocessing
- Unit tests for validation
- Integration tests for end-to-end workflows

## Contributing

1. Run tests: `pytest`
2. Check code quality: `ruff check titanic_ml/`
3. Format code: `black titanic_ml/`

## License

MIT License
>>>>>>> 7aeb567 (Initial commit: Titanic Random Forest + MLOps)
