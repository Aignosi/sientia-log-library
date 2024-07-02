import mlflow
from typing import Any, Dict, List

class BaseLog:
    """
    Basic Log object that don't have any requirements nor parameters to be used
    """
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

class SimpleLog(BaseLog):
    """
    Log object for generic models. Requires a dataset name and model inputs
    """
    def __init__(self, dataset_name: str, inputs: str | List[str] ) -> None:
        """
        params:
            dataset_name: Name of the dataset
            inputs: Name of the model inputs
        """
        self.dataset_name = dataset_name
        self.inputs = inputs


        mlflow.log_params({
            "Dataset": dataset_name,
            "Inputs": inputs,
            })