import mlflow
import mlflow.sklearn
from typing import Any
from sientia_tracker.model_serving.model_tracker import ModelTracker
from sientia.exceptions import SientiaMlException

class RetrieverTracker:
    def __init__(self, client):
        self.client = client
        self.model_tracker = ModelTracker(client)

    def get_artifact(
        self,
        destination: str,
        search_by: str,
        run_id: str | None = None,
        model_name: str | None = None,
        artifact_name: str | None = None,
    ) -> None | Any:
        """
        Get an artifact in MLflow by experiment or model and save it to a destination path.
        If the artifact is searched by model, the latest production version will be used.

        Args:
            destination: The destination path to save the artifact.
            search_by: The way to search for the artifact ('experiment' or 'model').
            run_id: The run ID of the experiment (if search_by is "experiment").
            model_name: The name of the model (if search_by is "model").
            artifact_name: The path of the artifact to download.

        Returns:
            artifact: The artifact(.csv) downloaded from MLflow.
        """

        if search_by == "experiment":
            # Check if artifact exists
            try:
                run_artifacts = self.client.list_artifacts(run_id=run_id)
            except SientiaMlException as e:
                print(f"Error: {e}")
                return None
            if run_artifacts is None:
                return None

            if artifact_name not in [artifact.path for artifact in run_artifacts]:
                print(
                    f'Artifact {artifact_name} not found in run {run_id}')
                return None
            try:
                artifact = self.client.download_artifacts(
                    run_id=run_id, path=artifact_name, dst_path=destination)
            except SientiaMlException as e:
                print(f"Error: {e}")
                return None
            
            return artifact

        elif search_by == "model":
            # Get the latest production version of the model
            try:
                latest_production_id = self.model_tracker.get_model_run_id(model_name)
                # Check if artifact exists
                run_artifacts = self.client.list_artifacts(
                    run_id=latest_production_id)
            except SientiaMlException as e:
                print(f"Error: {e}")
                return None
            
            if run_artifacts is None:
                return None
            
            if artifact_name not in [artifact.path for artifact in run_artifacts]:
                print(
                    f'Artifact {artifact_name} not found in run {latest_production_id}')
                return None
            try:
                artifact = self.client.download_artifacts(
                    run_id=latest_production_id,
                    path=artifact_name,
                    dst_path=destination,
                )
                return artifact
            except SientiaMlException as e:
                print(f"Error: {e}")
                return None
        else:
            print(
                'Invalid artifact type. Please specify either "experiment" or "model" as the artifact type.')
            return None

    def get_prediction(self, model_id, data, by: str = "model"):
        """
        Get a prediction from a model or run

        Args:
            id (str): The run ID or model name.
            data (pandas.DataFrame): The data to use for the prediction.
            by (str): The way to search for the model ("run" or "model").

        Returns:
            pandas.DataFrame: A DataFrame containing the predictions.
        """
        if by == "model":
            model_uri = f"models:/{model_id}/production"
        elif by == "run":
            model_uri = f"runs:/{model_id}/prediction_model"
        else:
            print(
                'Invalid artifact type. Please specify either "model" or "run" as the artifact type.'
            )
            return None
        model = mlflow.pyfunc.load_model(model_uri, suppress_warnings=True)
        predictions = model.predict(data)
        return predictions

    def get_transformed_data(self, model_id, data, by: str = "run"):
        """
        Get a transformed data from a model or run.

        Args:
            model_id (str): The run ID or model name.
            data (pandas.DataFrame): The data to use for the transformation.
            by (str): The way to search for the model ("run" or "model").

        Returns:
            pandas.DataFrame: A DataFrame containing the transformed data.
        """
        if by == "model":
            latest_production_id = self.model_tracker.get_model_run_id(
                model_name=model_id, stage="Production"
            )
            model_uri = self.model_tracker.get_model_uri(latest_production_id, prediction=False)
        elif by == "run":
            model_uri = self.model_tracker.get_model_uri(run_id=model_id, prediction=False)
        else:
            print(
                'Invalid artifact type. Please specify either "model" or "run" as the artifact type.'
            )
            return None

        model = mlflow.sklearn.load_model(model_uri)
        transformed_data = model.predict(data)
        """
        import  pdb
        pdb.set_trace()
        """
        return transformed_data
