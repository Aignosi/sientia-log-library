# SIENTIA™ Log Library

The SIENTIA™ Log Library is a Python library designed to facilitate tracking and logging for various data analysis and machine learning workflows. It provides a seamless integration with MLflow, enabling users to log experiments, models, metrics, and artifacts efficiently.

## Features

- **Base Tracker**: A generic tracker with no additional requirements.
- **Simple Tracker**: A lightweight tracker requiring only dataset and input names.
- **Regression Tracker**: A specialized tracker for regression models with support for logging model-specific parameters and metrics.
- **Integration with MLflow**: Log models, parameters, metrics, and artifacts directly to an MLflow server.

## Installation

To install the library, use the following command:

```bash
pip install sientia_tracker
```

## Documentation

Comprehensive documentation for the library is available [here](https://aignosi.github.io/sientia-log-library/) (Work in Progress).

## Quick Start

### Example: Using the Simple Tracker

```python
from sientia_tracker.simple_tracker import SimpleTracker

# Initialize tracking
tracking_uri = "http://localhost:5000"
username = "your_username"
password = "your_password"
project_name = "example_project"

tracker = SimpleTracker(tracking_uri, username, password)
tracker.set_project(project_name)

# Start a run
run = tracker.save_experiment(dataset_name="example_dataset", inputs=["feature1", "feature2"])
run_id = run.info.run_id

# Log parameters and metrics
tracker.log_params({"param1": "value1", "param2": "value2"})
tracker.log_metrics({"accuracy": 0.95})
```

## Requirements

The library requires the following dependencies:

- Python 3.7+
- `mlflow==2.10.1`
- `typing`

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Authors

- Ítalo Azevedo ([italo@aignosi.com.br](mailto:italo@aignosi.com.br))
- Pedro Bahia ([pedro.bahia@aignosi.com.br](mailto:pedro.bahia@aignosi.com.br))
- Matheus Demoner ([matheus@aignosi.com.br](mailto:matheus@aignosi.com.br))

---

For any questions or support, feel free to reach out to the authors or open an issue in the repository.