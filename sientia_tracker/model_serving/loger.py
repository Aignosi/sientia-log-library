import mlflow.sklearn
from typing import Any
import pandas as pd
import mlflow

class LogTracker:
    def log_prediction_model(self,sk_model: Any) -> None:
        """
            Log a prediction model. A prediction model is any sklearn regressor model.

            Args:
                sk_model: scikit-learn model to be saved.
                artifact_path: Run-relative artifact path.

            Returns:
                None
        """
        mlflow.sklearn.log_model(
            sk_model, 
            "prediction_model", 
        )

    def log_data_model(self,sk_model: Any) -> None:
        """
            Log a data model. A data model is any sklearn model whose aim
            is to process data before the calling of a prediction model.

            Args:
                sk_model: scikit-learn model to be saved.
                artifact_path: Run-relative artifact path.

            Returns:
                None
        """
        mlflow.sklearn.log_model(
            sk_model, 
            "data_model", 
        )

    def log_sientia_model(self, sk_model: Any, artifact_path: Any, **kwargs) -> None:
        """
        Log a SIENTIA model. A SIENTIA model is any sklearn model build through the 
        SIENTIA mlops library.

        Args:
            sk_model: scikit-learn model to be saved.
            artifact_path: Run-relative artifact path.

        Returns:
            None
        """
        mlflow.sklearn.log_model(
            sk_model, 
            artifact_path, 
            # mlops library pip requirement
            extra_pip_requirements=[
                "git+https://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@github.com/Aignosi/sientia-mlops-library.git"
            ], 
            **kwargs
        )

    def log_params(self, params: dict[str,Any]):
        """
        Log parameters to MLflow.

        Args:
            params: The parameters to log.

        Returns:
            None
        """
        mlflow.log_params(params)

    def log_param(self, key:str, value:Any) -> None:
        """
        Log a param in the active run.

        Args:
            key (str): Param name
            value (any): Param value
        
        Returns:
            None
        """
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str,float]):
        """
        Log metrics to MLflow.

        Args:
            metrics: The metrics to log.

        Returns:
            None
        """
        mlflow.log_metrics(metrics)
    
    def log_metric(self, key:str, value:Any) -> None:
        """
        Log a metric in the active run.

        Args:
            key (str): Metric name
            value (any): Metric value
        
        Returns:
            None
        """
        mlflow.log_metric(key, value)

    def log_artifact(self,local_path: str, artifact_path: str | None = None, run_id: str | None = None) -> None:
        """
        Log an artifact.

        Args:
            local_path: Local path of the artifact to log.
            artifact_path: If provided, the directory in artifact_uri to write to.
            run_id: optional id of current run

        Returns:
            None
        """
        mlflow.log_artifact(
            local_path=local_path,
            artifact_path=artifact_path,
            run_id=run_id
        )
    
    def log_inputs(self, dataset: pd.DataFrame, source: str | None = None, context: str | None = None, tags: dict[str, str] | None = None):
        dataset = mlflow.data.from_pandas(dataset, source=source)
        mlflow.log_input(
            dataset=dataset,
            context=context,
            tags=tags
        )