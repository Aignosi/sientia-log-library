from sientia_tracker.base_tracker import BaseTracker
import mlflow
from typing import List, Union


class SimpleTracker(BaseTracker):
    """
    A tracker object for generic models, designed to log experiments and parameters
    to the SIENTIA edge AIOps platform. This tracker requires minimal setup, making it
    ideal for users, including citizen data scientists, who want to track their workflows
    with ease.
    """

    def __init__(self, tracking_uri: str, username: str = None, password: str = None) -> None:
        """
        Initialize the SimpleTracker and connect to the SIENTIA platform.

        Parameters:
            tracking_uri (str): The URI of the SIENTIA platform (powered by MLflow API).
            username (str, optional): The username for accessing the SIENTIA platform (if required).
            password (str, optional): The password for accessing the SIENTIA platform (if required).

        Example:
            tracker = SimpleTracker("http://sientia.ai", username="user", password="pass")
        """
        super().__init__(tracking_uri, username, password)

    def save_experiment(self, dataset_name: str = "data", inputs: Union[str, List[str]] = "inputs") -> mlflow.ActiveRun:
        """
        Start a new experiment run on the SIENTIA platform.

        Parameters:
            dataset_name (str, optional): The name of the dataset used in the experiment. Default is "data".
            inputs (Union[str, List[str]], optional): The input feature(s) used in the model. Can be a single string
                or a list of strings. Default is "inputs".

        Returns:
            mlflow.ActiveRun: The active run object for the experiment.

        Example:
            tracker.save_experiment(
                dataset_name="customer_data",
                inputs=["age", "income", "purchase_history"]
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
            "Dataset": dataset_name,
            "Inputs": inputs,
        })

        return active_run
