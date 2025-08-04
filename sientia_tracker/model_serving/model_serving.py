import mlflow
import os

from sientia_tracker.model_serving.experiment_tracker import ExperimentTracker
from sientia_tracker.model_serving.loger import LogTracker
from sientia_tracker.model_serving.model_tracker import ModelTracker
from sientia_tracker.model_serving.retriever import RetrieverTracker
from sientia_tracker.model_serving.run_tracker import RunTracker

class ModelServing:
    def __init__(self, tracking_uri, username: str = None, password: str = None):
        # set tracking uri
        mlflow.set_tracking_uri(tracking_uri)

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        # Create an MLflow client
        self.client = mlflow.tracking.MlflowClient()
        
        self.LogTracker = LogTracker()
        self.RunTracker = RunTracker()
        self.ExperimentTracker = ExperimentTracker()
        self.ModelTracker = ModelTracker(self.client)
        self.RetrieverTracker = RetrieverTracker(self.client)
        
        # Enable legacy suport so outdated code can still work with this new interface
        self._delegated_objects = []
        self._delegated_objects.append(self.LogTracker)
        self._delegated_objects.append(self.RunTracker)
        self._delegated_objects.append(self.ExperimentTracker)
        self._delegated_objects.append(self.ModelTracker)
        self._delegated_objects.append(self.RetrieverTracker)

        
    def __getattr__(self, nome):
        for obj in self._delegated_objects:
            try:
                attr = getattr(obj, nome)
                setattr(self, nome, attr)
                return attr
            except AttributeError:
                continue
        raise AttributeError(f"'{self.__class__.__name__}' não tem o atributo '{nome}'")