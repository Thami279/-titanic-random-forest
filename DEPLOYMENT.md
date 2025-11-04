# Deployment Guide

## Overview

This guide covers deploying the Titanic survival prediction model to production, including batch and online scoring, containerization, and monitoring.

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- MLflow tracking server (optional but recommended)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or with MLflow support:
pip install -r requirements.txt mlflow
```

### 2. Training Pipeline

Train and register models using MLflow:

```python
from titanic_ml.models.tracking import ExperimentTracker
from titanic_ml.utils.preprocessing import build_preprocessor, engineer_features
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data
df = pd.read_csv('titanic.csv')
engineered = engineer_features(df)
# ... feature selection ...

# Train model
tracker = ExperimentTracker(use_mlflow=True, experiment_name="titanic_prod")
# ... training code ...

# Model is automatically logged to MLflow
```

### 3. Model Registry

Models are registered in MLflow. Access via:

```python
import mlflow

# Load best model
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## Deployment Options

### Option 1: Batch Scoring

For batch processing of CSV files:

```python
#!/usr/bin/env python3
"""Batch scoring script."""

import pandas as pd
import mlflow.sklearn
import sys

def batch_score(input_csv: str, output_csv: str, model_uri: str):
    """Score a batch of records."""
    df = pd.read_csv(input_csv)
    
    # Load model
    model = mlflow.sklearn.load_model(model_uri)
    
    # Preprocess and predict
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    
    # Save results
    df['prediction'] = predictions
    df['probability'] = probabilities
    df.to_csv(output_csv, index=False)
    
    print(f"Scored {len(df)} records, saved to {output_csv}")

if __name__ == "__main__":
    batch_score(sys.argv[1], sys.argv[2], sys.argv[3])
```

Usage:
```bash
python batch_score.py input.csv output.csv "runs:/<run_id>/model"
```

### Option 2: Online Scoring API (Flask)

Simple REST API for real-time predictions:

```python
from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)
model = mlflow.sklearn.load_model("runs:/<run_id>/model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]
    
    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 3: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_ml/ ./titanic_ml/
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t titanic-ml:latest .
docker run -p 5000:5000 titanic-ml:latest
```

## Monitoring

### Data Validation

Implement data validation checks:

```python
from titanic_ml.utils.validation import validate_data

def validate_input(data: dict) -> bool:
    """Validate input data before prediction."""
    df = pd.DataFrame([data])
    
    results = validate_data(
        df,
        required_columns=['Pclass', 'Sex', 'Age', 'Fare'],
        numeric_ranges={'Age': (0, 120), 'Fare': (0, 1000)},
        categorical_values={'Sex': ['male', 'female']}
    )
    
    return all(results.values())
```

### Performance Monitoring

Track model performance over time:

1. **Prediction Logging**: Log all predictions with timestamps
2. **Drift Detection**: Monitor input feature distributions
3. **Performance Metrics**: Track accuracy on labeled data
4. **Alerting**: Set up alerts for performance degradation

### Model Retraining

Schedule periodic retraining:

```bash
# Run via cron or scheduler
0 0 * * 0  # Weekly retraining
cd /path/to/project
python train.py --experiment-name titanic_prod
```

## CI/CD Pipeline

### GitHub Actions Example

`.github/workflows/train.yml`:

```yaml
name: Train Model

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python train.py
      - name: Deploy model
        if: success()
        run: |
          # Deploy to production
          python deploy.py --model-uri ${{ env.MODEL_URI }}
```

## Testing in Production

### Smoke Tests

```python
def test_model_serving():
    """Smoke test for model serving."""
    response = requests.post('http://localhost:5000/predict', json={
        'Pclass': '1', 'Sex': 'female', 'Age': 30, 'Fare': 50.0
    })
    assert response.status_code == 200
    assert 'prediction' in response.json()
```

### Performance Budgets

Define acceptable performance thresholds:

- **Accuracy**: > 0.75 on held-out test set
- **Latency**: < 100ms per prediction
- **Throughput**: > 100 predictions/second

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check MLflow URI and model version
2. **Low accuracy**: Monitor for data drift, consider retraining
3. **High latency**: Optimize feature engineering or use faster model variant

### Rollback Procedure

```bash
# Revert to previous model version
mlflow models serve -m "models:/titanic_survival/previous" -p 5000
```

## Next Steps

- Set up automated monitoring dashboard
- Implement A/B testing framework
- Create model versioning strategy
- Document data pipeline and feature store

