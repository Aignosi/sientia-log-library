from typing import Any
import pandas as pd
import mlflow

class ExperimentTracker:
    def search_experiments(self):
        """
        Search and retrieve a list of MLflow experiments.

        Returns:
            pandas.DataFrame: A DataFrame containing experiment information.
        """
        experiments = mlflow.search_experiments()
        return experiments
    
    def set_experiment(self, experiment_identifier:str) -> None:
        """
        Set the given experiment as the active experiment.

        Args:
            experiment_identifier (str): name or id of the experiment to be setted
        """
        mlflow.set_experiment(experiment_identifier)
    
    def create_set_experiment(self, experiment_name:str, tags: None | dict[str,Any] = None) -> None:
        """
            Create a new experiment with name experiment_name and set to active. 
            If the experiment already exists, it will only set it to active and nothing is created.

            Args:
                experiment_name (str): Name of the experiment 
                tags (dict[str, Any]): An optional dictionary of string keys and values to set as tags on the experiment.
        """
        self.experiment_name = experiment_name
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name,tags=tags)
        mlflow.set_experiment(experiment_name)

    def get_experiments(self,experiments_name: list[str], order_by: None|list[str] = None) -> pd.DataFrame:
        """
        Return the MLflow experiments whose name is in experiments_name.

        Args:
            experiments_name (list[str]): List with experiments to search for.

        Returns:
            pandas.DataFrame: A DataFrame containing experiment information.
        """
        experiments = mlflow.search_experiments(experiments_name=experiments_name, order_by=order_by)
        return experiments

    def list_experiments(self):
        """
        List all available MLflow experiments.

        Returns:
            pandas.DataFrame: A DataFrame containing experiment information.
        """
        experiments = mlflow.search_experiments()
        return experiments
    
    def save_experiment(self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, Any] | None = None,
        description: str | None = None,
        log_system_metrics: bool | None = None,
    ) -> mlflow.ActiveRun:
        """
        Save a experiment.

        Args:
            run_id: If specified, get the run with the specified UUID and log parameters and metrics under that run. 
            experiment_id: ID of the experiment under which to create the current run (applicable only when run_id is not specified).
            run_name: Name of new run. Used only when run_id is unspecified.
            nested: Controls whether run is nested in parent run. True creates a nested run.
            tags: An optional dictionary of string keys and values to set as tags on the run. If a run is being resumed, these tags are set on the resumed run. If a new run is being created, these tags are set on the new run.
            description: An optional string that populates the description box of the run.
            log_system_metrics: If True, system metrics will be logged. If None, we will check environment variable

        Returns:
            ActiveRun: object that acts as a context manager wrapping the run's state.
        """
        if run_name is None:
            all_runs = mlflow.search_runs(experiment_names=[self.experiment_name], order_by=["start_time desc"])

            if nested==False:
                next_run_number = len(all_runs) + 1
                run_name = f"{self.experiment_name}-{next_run_number}"
            else:
                active_run = mlflow.active_run()
                active_run_id = active_run.data.tags.get("mlflow.parentRunId")

                parent_run_id = active_run_id if active_run_id is not None else active_run.info.run_id
                parent_run_name = mlflow.get_run(parent_run_id).info.run_name
                try:
                    child_runs = all_runs[all_runs["tags.mlflow.parentRunId"] == parent_run_id]
                except KeyError:
                    child_runs = []
                next_run_number = len(child_runs) + 1
                run_name = f"{parent_run_name}-{next_run_number}"

        run = mlflow.start_run(
            run_id = run_id,
            experiment_id = experiment_id,
            run_name = run_name,
            nested = nested,
            tags = tags,
            description = description,
            log_system_metrics= log_system_metrics
        )
        return run