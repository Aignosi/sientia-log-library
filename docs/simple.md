# Simple Tracker

## Overview

The `SimpleTracker` class, which inherits from `BaseTracker`, is designed to provide a minimal and user-friendly way to track models. It requires only the specification of a dataset name and input feature names, making it ideal for users who want a straightforward and hassle-free tracking experience.

This tracker is perfect for citizen data scientists and other users who need a simple yet effective tool to log their experiments on the **SIENTIA edge AIOps platform**.

## Key Features

- **Minimal Setup**: Requires only basic information like dataset name and input features
- **User-Friendly Interface**: Simplified API for easy adoption by non-technical users
- **Seamless Integration**: Built on the BaseTracker foundation for consistent experience
- **Automatic Metadata**: Captures essential information without manual configuration

## Usage Examples

### Basic Usage

```python
# Import the SimpleTracker class
from sientia_tracker.simple import SimpleTracker

# Initialize the tracker with your SIENTIA platform URL and credentials
tracker = SimpleTracker("http://sientia.ai", username="user", password="pass")

# Set up a project to organize your experiments
tracker.set_project("customer_analysis")

# Start an experiment with dataset and input features
tracker.save_experiment(
    dataset_name="customer_data",
    inputs=["age", "income", "purchase_history"]
)

# Log additional metrics from your model
tracker.log_metrics({"accuracy": 0.92, "precision": 0.88})
```

### Advanced Usage

```python
# Import necessary libraries
from sientia_tracker.simple import SimpleTracker
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Initialize the tracker
tracker = SimpleTracker("http://sientia.ai", username="user", password="pass")
tracker.set_project("customer_churn_prediction")

# Prepare your data (example)
df = pd.read_csv("customer_data.csv")
X = df[['age', 'tenure', 'monthly_charges']]
y = df['churn']

# Start tracking experiment with dataset information
tracker.save_experiment(
    dataset_name="telco_customer_churn",
    inputs=['age', 'tenure', 'monthly_charges']
)

# Train your model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Log model performance metrics
tracker.log_metrics({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "sample_count": len(X)
})

# Log model parameters
tracker.log_params({
    "n_estimators": 100,
    "max_depth": model.max_depth,
    "algorithm": "RandomForest"
})

# Save the experiment
tracker.end_run()
```

---

## API Reference

The following section contains the detailed API reference for the SimpleTracker class.

!!! note "Method Documentation"
    Each method is documented with its parameters, return values, and examples.
    Methods are separated by horizontal rules for better readability.

::: sientia_tracker.simple_tracker
    options:
        show_root_heading: true
        show_category_heading: true
        heading_level: 2
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        docstring_section_style: spacy