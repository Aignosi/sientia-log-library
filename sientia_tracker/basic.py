import mlflow
import os
import mlflow.sklearn
from typing import Any, Dict

# remove warnings
import warnings
warnings.filterwarnings('ignore')

class BaseTracker:
    """
    Basic Tracker object that don't have any requirements nor parameters to be used
    """
    def __init__(self, tracking_uri, username: str = None, password: str = None):
        # set tracking uri
        mlflow.set_tracking_uri(tracking_uri)

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        # Create an MLflow client
        self.client = mlflow.tracking.MlflowClient()

    def log_model(self, sk_model: Any, artifact_path: Any, extra_pip_requirements: Any | None = None,**kwargs):
        """
        Log a model to MLflow.

        Args:
            sk_model: scikit-learn model to be saved
            artifact_path: name of the model

        Returns:
            None
        """
        mlflow.sklearn.log_model(sk_model, artifact_path, extra_pip_requirements,**kwargs)

    def log_params(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dict with the parameters to log.

        Returns:
            None
        """
        mlflow.log_params(params, **kwargs)

    def log_metrics(self, params: Dict[str, float], **kwargs) -> None:
        """
        Log metrics to MLflow.

        Args:
            params: Dict with the metrics to log.

        Returns:
            None
        """
        mlflow.log_metrics(params, **kwargs)

    def log_artifact(self, local_path: str, artifact_path: str | None = None, **kwargs) -> None:
        """
        Log an artifact to MLflow.

        Args:
            local_path: Path to the file to write.
            artifact_path: If provided, the directory in artifact_uri to write to.

        Returns:
            None
        """
        mlflow.log_artifact(local_path, artifact_path, **kwargs)

    def set_project(self, project_name: str):
         # Check if the experiment already exists; if not, create it
        project = mlflow.get_experiment_by_name(
            project_name)
        if project is None:
            mlflow.create_experiment(name=project_name)
        else:
            # Activate the experiment for tracking
            mlflow.set_experiment(project_name)
            self.project_name = project_name
            print(f"Experiment {project_name} already exists")

    def save_experiment(self):
        """
        Start a run in MLflow.

        Args:
            **kwargs: The parameters to log.

        Returns:
            None
        """
        print("Saving experiment", self.project_name)
        runs = mlflow.search_runs(experiment_names=[
            self.project_name], order_by=["start_time desc"])
        next_run_number = len(runs) + 1
        active_run = mlflow.start_run(run_name=f"{self.project_name}-{next_run_number}")
        return active_run

    def get_model_run_id(self, model_name: str, stage: str = "Production"):
        """
        Get the run_id of a model based on its name and stage.

        Args:
            model_name (str): The name of the model.
            stage (str): The stage of the model.

        Returns:
            str: The run_id of the model.
        """
        latest_versions = self.client.get_latest_versions(
            name=model_name, stages=[stage])
        run_id = latest_versions[0].source.split("/")
        return run_id[2]

    def get_model_experiment_id(self, model_name: str):
        """
        Get the project name associated with a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: The project name.
        """
        latest_production_id = self.get_model_run_id(model_name=model_name, stage="Production")
        run_info = mlflow.get_run(latest_production_id)
        return run_info.info.experiment_id
    
    def get_run_name(self, run_id: str):
        """
        Get the run name associated with a run ID.

        Args:
            run_id (str): The run ID.

        Returns:
            str: The run name.
        """
        if run_id:
            run_info = mlflow.get_run(run_id)
            run_name = run_info.info.run_name
        else:
            run_name = None
        return run_name
