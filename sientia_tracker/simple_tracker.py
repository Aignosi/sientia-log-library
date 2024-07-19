from sientia_tracker.basic import BaseTracker
import mlflow
from typing import List


class SimpleTracker(BaseTracker):
    """
    Tracker object for generic models. Requires a dataset name and model inputs
    """
    def __init__(self, tracking_uri, username: str = None, password: str = None):
        super().__init__(tracking_uri, username, password)

    def save_experiment(self, dataset_name: str = "data", inputs: str | List[str] = "inputs") -> mlflow.ActiveRun:
        """
        Start a run in MLflow.

        params:
            dataset_name: Name of the dataset
            inputs: Name of the model inputs
        """

        mlflow.end_run()
        print("Saving experiment", self.project_name)
        runs = mlflow.search_runs(experiment_names=[
            self.project_name], order_by=["start_time desc"])
        next_run_number = len(runs) + 1
        active_run = mlflow.start_run(run_name=f"{self.project_name}-{next_run_number}")
        mlflow.log_params({
            "Dataset": dataset_name,
            "Inputs": inputs,
            })
        return active_run

