## Regression

### Overview

The `RegressionTracker` class, which extends the `BaseTracker`, is specifically designed to help users track regression models with ease. It simplifies the process of managing regression workflows by allowing users to log key components such as variable names, model hyperparameters, train/test split configurations, and the regression model itself.

This tracker is ideal for users who want a structured and intuitive way to monitor and log their regression experiments on the **SIENTIA edge AIOps platform**.

### Key Features

- **Regression-Specific Logging**: Tailored for tracking regression models and their performance metrics
- **Comprehensive Metadata**: Captures dataset details, feature information, and model configuration
- **Performance Tracking**: Built-in support for logging R² scores and other regression metrics
- **Flexible Configuration**: Supports various training configurations like train/test split and data shuffling

### Usage Examples

```python
# Initialize the regression tracker
tracker = RegressionTracker("http://sientia-platform.com", username="user", password="pass")

# Set up a project
tracker.set_project("sales_forecasting")

# Start an experiment with a regression model and related parameters
tracker.save_experiment(
    model=linear_model,
    model_name="linear_regression",
    dataset_name="sales_data",
    inputs=["season", "promotion", "price"],
    target="sales",
    date_column="transaction_date",
    r2=0.85,
    train_size=0.75,
    shuffle=True
)

# Log additional metrics if needed
tracker.log_metrics({"mean_absolute_error": 0.15, "mean_squared_error": 0.05})
```

::: sientia_tracker.regression
