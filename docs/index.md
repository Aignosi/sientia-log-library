# SIENTIA™ Log Library Documentation

![SIENTIA Logo](img/sientia_logo.png)

Welcome to the documentation for the **SIENTIA™ Log Library**! This library is designed to simplify tracking and logging for a wide range of data analysis and machine learning workflows. Whether you're a data scientist, machine learning engineer, or a citizen data scientist, this tool provides an intuitive way to manage experiments, models, and metrics.

## Overview

The SIENTIA™ Log Library is built on top of MLflow and provides a streamlined interface for tracking machine learning experiments. It offers several key benefits:

- **Simplified Tracking**: Easy-to-use API for logging models, parameters, and metrics
- **Consistent Structure**: Standardized approach to experiment organization
- **Flexible Integration**: Works with various ML frameworks and tools
- **Scalable Architecture**: Designed to handle projects of any size
- **User-Friendly**: Accessible for both technical and non-technical users

## Why Use SIENTIA™ Log Library?

In today's fast-paced data science environment, keeping track of experiments is crucial. The SIENTIA™ Log Library addresses common challenges:

| Challenge | Solution |
|-----------|----------|
| **Experiment Reproducibility** | Automatically tracks code versions, parameters, and dependencies |
| **Model Comparison** | Standardized metrics logging for easy comparison across experiments |
| **Team Collaboration** | Centralized tracking server for sharing results with team members |
| **Deployment Readiness** | Structured metadata to streamline the path from experiment to production |

## Modules

Explore the key components of the library:

* [Basic Tracker](basic.md): A generic tracker for logging experiments with minimal setup, providing core functionality for all tracking needs.
* [Simple Tracker](simple.md): A lightweight tracker for logging datasets and input features with ease, ideal for straightforward ML workflows.
* [Regression Tracker](regression.md): A specialized tracker for regression models, including support for logging model-specific metrics and parameters like R² scores and train/test configurations.

## Quick Start

Get started quickly with these interactive examples:

* [Simple Tracker Notebook](simple_tracker.ipynb): Learn how to use the Simple Tracker to log datasets and features, with step-by-step examples.
* [Regression Tracker Notebook](regression_tracker.ipynb): Explore how to track regression models and their metrics, with practical demonstrations.

## Installation

```bash
pip install sientia-log-library
```

## Basic Usage

```python
from sientia_tracker.basic import BaseTracker

# Initialize the tracker
tracker = BaseTracker("http://sientia-platform.com", username="user", password="pass")

# Set up a project
tracker.set_project("my_project")

# Log parameters and metrics
tracker.log_params({"learning_rate": 0.01, "batch_size": 32})
tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
```

## Getting Support

If you encounter any issues or have questions about using the SIENTIA™ Log Library, please reach out to our support team at support@sientia.ai or visit our [community forum](https://community.sientia.ai).