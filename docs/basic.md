## Basic

### Overview

The `BaseTracker` class in the `sientia_tracker.basic` module provides a comprehensive tracker for logging experiments, models, parameters, and metrics to the **SIENTIA edge AIOps platform** (powered by the MLflow API). It is designed as a simple interface for users to manage their machine learning workflows without requiring additional setup or parameters.

### Key Features

- **Simple Setup**: Initialize with just a tracking URI and optional authentication
- **Comprehensive Logging**: Log models, parameters, metrics, and artifacts
- **Project Management**: Create and manage experiments
- **Model Tracking**: Retrieve model information by name and stage

### Usage Examples

```python
# Initialize the tracker
tracker = BaseTracker("http://sientia-platform.com", username="user", password="pass")

# Set up a project
tracker.set_project("my_ml_project")

# Log parameters
tracker.log_params({"learning_rate": 0.01, "batch_size": 32})

# Log metrics
tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})

# Log a model
tracker.log_model(model, "my_model", extra_pip_requirements=["numpy==1.21.0"])
```

---

## API Reference

The following section contains the detailed API reference for the BaseTracker class.

!!! note "Method Documentation"
    Each method is documented with its parameters, return values, and examples.
    Methods are separated by horizontal rules for better readability.

::: sientia_tracker.basic
    options:
        show_root_heading: true
        show_category_heading: true
        heading_level: 2
        members_order: source
        separate_signature: true
        show_signature_annotations: true
        docstring_section_style: spacy
