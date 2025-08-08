import mlflow
from typing import Any, List, Union
from sientia_tracker.logs.base_tracker import BaseTracker

class RegressionTracker(BaseTracker):
    """
    A specialized tracker for regression models, designed to log experiments, models, and metrics
    to the SIENTIA edge AIOps platform. This tracker simplifies the process of managing regression
    workflows for users, including citizen data scientists.
    """

    def __init__(self, tracking_uri: str, username: str = None, password: str = None) -> None:
        """
        Initialize the RegressionTracker and connect to the SIENTIA platform.

        Parameters:
            tracking_uri (str): The URI of the SIENTIA platform (powered by MLflow API).
            username (str, optional): The username for accessing the SIENTIA platform (if required).
            password (str, optional): The password for accessing the SIENTIA platform (if required).

        Example:
            tracker = RegressionTracker("http://sientia-platform.com", username="user", password="pass")
        """
        super().__init__(tracking_uri, username, password)

    def save_experiment(self,
                        model: Any,
                        model_name: str = "regr_model",
                        dataset_name: str = "data",
                        inputs: Union[str, List[str]] = "inputs",
                        target: str = "target",
                        date_column: str = "date",
                        r2: float = 1.0,
                        train_size: float = 0.8,
                        shuffle: bool = True
                        ) -> mlflow.ActiveRun:
        """
        Start a new experiment run on the SIENTIA platform for a regression model.

        Parameters:
            model (Any): The regression model (e.g., Scikit-Learn model) to log.
            model_name (str, optional): The name of the model. Default is "regr_model".
            dataset_name (str, optional): The name of the dataset used for training. Default is "data".
            inputs (Union[str, List[str]], optional): The input feature(s) used for training. Default is "inputs".
            target (str, optional): The target variable for the regression task. Default is "target".
            date_column (str, optional): The name of the date column in the dataset. Default is "date".
            r2 (float, optional): The R² score of the model. Default is 1.0.
            train_size (float, optional): The proportion of the dataset used for training. Default is 0.8.
            shuffle (bool, optional): Whether the training data was shuffled. Default is True.

        Returns:
            mlflow.ActiveRun: The active run object for the experiment.

        Example:
            tracker.save_experiment(
                model=linear_model,
                model_name="linear_regression",
                dataset_name="sales_data",
                inputs=["feature1", "feature2"],
                target="sales",
                date_column="date",
                r2=0.85,
                train_size=0.75,
                shuffle=True
            )
        """
        # End any existing run
        mlflow.end_run()

        print(f"Saving experiment '{self.project_name}' on the SIENTIA platform...")
        # Retrieve existing runs and determine the next run number
        runs = mlflow.search_runs(experiment_names=[self.project_name], order_by=["start_time desc"])
        next_run_number = len(runs) + 1

        # Start a new run
        active_run = mlflow.start_run(run_name=f"{self.project_name}-{next_run_number}")

        # Log parameters
        mlflow.log_params({
            "Model Name": model_name,
            "Dataset Name": dataset_name,
            "Inputs": inputs,
            "Target": target,
            "Date Column": date_column,
            "Train Size": train_size,
            "Shuffle": shuffle,
        })

        # Log the regression model
        mlflow.sklearn.log_model(model, model_name)

        # Log metrics
        mlflow.log_metrics({"R² Score": r2})

        return active_run