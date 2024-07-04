
import mlflow
from typing import Any, List
from basic import BaseTracker

class RegressionTracker(BaseTracker):
    """
    Tracker object for regression models.
    """
    def __init__(self, tracking_uri, username: str = None, password: str = None):
        super().__init__(tracking_uri, username, password)

    def save_experiment(self,model: Any,
                    model_name: str = "regr_model",
                    dataset_name: str = "data",
                    inputs: str | List[str] = "inputs",
                    target: str = 'target',
                    date_column: str = "date",
                    r2: float = 1.0, 
                    train_size: float = 0.8,
                    shuffle: bool = True) -> mlflow.ActiveRun:
        """
        Start a run in MLflow.

        params:
            model: Scikit-Learn Regression Model
            model_name: The name of the model as defined by user. Default: regr_model
            dataset_name: The name of the dataset. Default: data
            inputs: The name of the inputs. Default: inputs
            target: The name of the target. Default: target
            date_column: The name of the date column. Default: date
            r2: r2-score of the model. Default: 1
            train_size: Proportion of the data used to traind the model. Default: 0.8
            shuffle: Wheter or not the train data was shuffled. Default: True
        """

        mlflow.log_params({
            "Model": "Linear Regression",
            "Dataset": dataset_name,
            "Inputs": inputs,
            "Date Column": date_column,
            "Target": target,
            "Train Size": train_size,
            "Shuffle": shuffle,
        })
        
        mlflow.log_model(model,model_name)
        mlflow.log_metrics({"r2":r2})
        
        print("Saving experiment", self.project_name)
        runs = mlflow.search_runs(experiment_names=[
            self.project_name], order_by=["start_time desc"])
        next_run_number = len(runs) + 1
        active_run = mlflow.start_run(run_name=f"{self.project_name}-{next_run_number}")
        return active_run