#TODO: Explicar o que é o basic tracker

import mlflow
import os
import mlflow.sklearn
from typing import Any, Dict, Optional

# remove warnings
import warnings
warnings.filterwarnings('ignore')

class BaseTracker:
    """
    A basic tracker for logging experiments, models, parameters, and metrics to the SIENTIA edge AIOps platform.
    This tracker provides a simple interface for users to manage their machine learning workflows.
    """

    def __init__(self, tracking_uri: str, username: Optional[str] = None, password: Optional[str] = None) -> None:
        """
        Initialize the tracker object and connect to the SIENTIA platform.

        Parameters:
            tracking_uri (str): The URI of the SIENTIA platform (powered by MLflow API).
            username (Optional[str]): The username for accessing the SIENTIA platform (if required).
            password (Optional[str]): The password for accessing the SIENTIA platform (if required).

        Example:
            tracker = BaseTracker("http://sientia-platform.com", username="user", password="pass")
        """
        mlflow.set_tracking_uri(tracking_uri)

        os.environ['MLFLOW_TRACKING_USERNAME'] = username or ""
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password or ""
        # Create an MLflow client
        self.client = mlflow.tracking.MlflowClient()

    def log_model(self, sk_model: Any, artifact_path: str, extra_pip_requirements: Optional[list[str]] = None, **kwargs) -> None:
        """
        Log a machine learning model to the SIENTIA platform.

        Parameters:
            sk_model (Any): The machine learning model (e.g., scikit-learn model) to log.
            artifact_path (str): The name or path where the model will be saved.
            extra_pip_requirements (Optional[list[str]]): Additional Python packages required to use the model.

        Example:
            tracker.log_model(model, "my_model", extra_pip_requirements=["numpy==1.21.0"])
        """
        mlflow.sklearn.log_model(sk_model, artifact_path, extra_pip_requirements, **kwargs)

    #TODO: Explicitar que estamos falando de hiperparâmetros
    def log_params(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Log a set of parameters to the SIENTIA platform.

        Parameters:
            params (Dict[str, Any]): A dictionary of parameter names and their values.

        Example:
            tracker.log_params({"learning_rate": 0.01, "batch_size": 32})
        """
        mlflow.log_params(params, **kwargs)

    def log_metrics(self, metrics: Dict[str, float], **kwargs) -> None:
        """
        Log a set of metrics to the SIENTIA platform.

        Parameters:
            metrics (Dict[str, float]): A dictionary of metric names and their values.

        Example:
            tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
        """
        mlflow.log_metrics(metrics, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None, **kwargs) -> None:
        """
        Upload a file or artifact to the SIENTIA platform.

        Parameters:
            local_path (str): The path to the file to upload.
            artifact_path (Optional[str]): The directory in the SIENTIA platform where the file will be stored.

        Example:
            tracker.log_artifact("results.csv", artifact_path="data")
        """
        mlflow.log_artifact(local_path, artifact_path, **kwargs)

    def set_project(self, project_name: str) -> None:
        """
        Set or create a project (experiment) on the SIENTIA platform.

        Parameters:
            project_name (str): The name of the project or experiment.

        Example:
            tracker.set_project("my_project")
        """
        project = mlflow.get_experiment_by_name(project_name)
        if project is None:
            mlflow.create_experiment(name=project_name)
            self.project_name = project_name
            print(f"Project '{project_name}' created on the SIENTIA platform.")
        else:
            mlflow.set_experiment(project_name)
            self.project_name = project_name
            print(f"Project '{project_name}' already exists on the SIENTIA platform.")

    def save_experiment(self) -> mlflow.ActiveRun:
        """
        Start a new run within the current project on the SIENTIA platform.

        Returns:
            mlflow.ActiveRun: The active run object for the experiment.

        Example:
            run = tracker.save_experiment()
        """
        print(f"Saving experiment '{self.project_name}' on the SIENTIA platform...")
        runs = mlflow.search_runs(experiment_names=[self.project_name], order_by=["start_time desc"])
        next_run_number = len(runs) + 1
        active_run = mlflow.start_run(run_name=f"{self.project_name}-{next_run_number}")
        return active_run

    def get_model_run_id(self, model_name: str, stage: str = "Production") -> str:
        """
        Retrieve the run ID of a model based on its name and stage from the SIENTIA platform.

        Parameters:
            model_name (str): The name of the model.
            stage (str): The stage of the model (e.g., "Production", "Staging").

        Returns:
            str: The run ID of the model.

        Example:
            run_id = tracker.get_model_run_id("my_model", stage="Production")
        """
        latest_versions = self.client.get_latest_versions(name=model_name, stages=[stage])
        run_id = latest_versions[0].source.split("/")
        return run_id[2]

    def get_model_experiment_id(self, model_name: str) -> str:
        """
        Retrieve the experiment ID associated with a model from the SIENTIA platform.

        Parameters:
            model_name (str): The name of the model.

        Returns:
            str: The experiment ID.

        Example:
            experiment_id = tracker.get_model_experiment_id("my_model")
        """
        latest_production_id = self.get_model_run_id(model_name=model_name, stage="Production")
        run_info = mlflow.get_run(latest_production_id)
        return run_info.info.experiment_id

    def get_run_name(self, run_id: str) -> Optional[str]:
        """
        Retrieve the name of a run based on its run ID from the SIENTIA platform.

        Parameters:
            run_id (str): The ID of the run.

        Returns:
            Optional[str]: The name of the run, or None if the run ID is invalid.

        Example:
            run_name = tracker.get_run_name("12345")
        """
        if run_id:
            run_info = mlflow.get_run(run_id)
            run_name = run_info.info.run_name
        else:
            run_name = None
        return run_name