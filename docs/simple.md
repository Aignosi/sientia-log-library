## Simple Tracker

### Overview

The `SimpleTracker` class, which inherits from `BaseTracker`, is designed to provide a minimal and user-friendly way to track models. It requires only the specification of a dataset name and input feature names, making it ideal for users who want a straightforward and hassle-free tracking experience.

This tracker is perfect for citizen data scientists and other users who need a simple yet effective tool to log their experiments on the **SIENTIA edge AIOps platform**.

### Key Features

- **Minimal Setup**: Requires only basic information like dataset name and input features
- **User-Friendly Interface**: Simplified API for easy adoption by non-technical users
- **Seamless Integration**: Built on the BaseTracker foundation for consistent experience

### Usage Examples

```python
# Initialize the tracker
tracker = SimpleTracker("http://sientia.ai", username="user", password="pass")

# Set up a project
tracker.set_project("customer_analysis")

# Start an experiment with dataset and input features
tracker.save_experiment(
    dataset_name="customer_data",
    inputs=["age", "income", "purchase_history"]
)

# Log additional metrics
tracker.log_metrics({"accuracy": 0.92, "precision": 0.88})
```

::: sientia_tracker.simple_tracker