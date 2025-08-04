
import os
import yaml
from sientia_tracker.model_serving.model_serving import ModelServing
# remove warnings
import warnings

warnings.filterwarnings("ignore")

import logging
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sientia.reports import Reports
from sientia.metrics import r2, mse, rce_drift
from sientia.exceptions import SientiaMlException
from typing import Any
import pandas as pd
import numpy as np
import mlflow
import os

class ModelServing:
    def __init__(self, tracking_uri, username: str = None, password: str = None):
        # set tracking uri
        mlflow.set_tracking_uri(tracking_uri)

        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        # Create an MLflow client
        self.client = mlflow.tracking.MlflowClient()

    # Function to get the model URI based on run_id

    def get_model_uri(self, run_id: str, prediction: bool = True):
        """
        Get the model URI based on the run_id.

        Args:
            run_id (str): The run_id of the model.

        Returns:
            str: The model URI.
        """
        run_info = mlflow.get_run(run_id)
        if prediction:
            model_uri = run_info.info.artifact_uri + "/prediction_model"
        else:
            model_uri = run_info.info.artifact_uri + "/data_model"
        return model_uri

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
            name=model_name, stages=[stage]
        )
        if not latest_versions:
            raise SientiaMlException(
                f"Model '{model_name}' in stage '{stage}' not found in the Model Registry."
            )
        else:
            run_id = latest_versions[0].source.split("/")
            return run_id[2]

    # Function to list runs for a given experiment
    def search_runs_by_name(self, experiment_names: list[str], order_by: None | list[str]=None) -> pd.DataFrame:
        """
        List runs for a specified MLflow experiment.

        Args:
            experiment_names (list[str]): List with experiment_names to retrieve runs from.

        Returns:
            pandas.DataFrame: A DataFrame containing run information.
        
        Raise:
            SientiaMlException if unable to search runs
        """
        try:
            runs = mlflow.search_runs(experiment_names=experiment_names,order_by=order_by)
        except SientiaMlException as e:
            logging.error(e)
            raise SientiaMlException
        return runs

    # Function to get information about registered models
    def get_models(self):
        """
        Get information about registered models.

        Returns:
            pandas.DataFrame: A DataFrame containing model information.
        """
        registered_models = mlflow.search_registered_models()
        model_data = []

        for rm in registered_models:
            model_info = dict(rm)
            model_data.append(model_info)

        df_model_info = pd.DataFrame(model_data)
        return df_model_info

    # Function to get information about model versions

    def models_info(self, model_name: list):
        """
        Get information about model versions for a given model name.

        Args:
            model_name (list): The name of the model.

        Returns:
            pandas.DataFrame: A DataFrame containing model version information.
        """
        try:
            mv = self.client.get_latest_versions(model_name)
        except SientiaMlException as e:
            logging.error(e)
            raise SientiaMlException
        
        try:
            versions_data = [
                {
                    "version": m.version,
                    "current_stage": m.current_stage,
                    "run_id": m.run_id,
                    "tags": m.tags.get("tags", []),
                    "name": model_name,
                }
                for m in mv
            ]
        except TypeError as e:
            logging.error(e)
        
        df = pd.DataFrame(versions_data)
        if not df.empty:
            max_version = int(df['version'].max())

        versions_df = pd.DataFrame()

        for i in range(1, max_version+1):
            try:
                version_info = pd.DataFrame(
                    self.client.get_model_version(model_name, i))
                version_info.set_index(0, inplace=True)
                versions_df = pd.concat(
                    [versions_df, version_info], axis=1, ignore_index=True)
            # Didn't specified the exceptions because we just wanna pass it
            except Exception:
                pass

        versions_df = versions_df.T
        try:
            versions_df = versions_df[[
                'version', 'current_stage', 'run_id', 'tags', 'name']]
            versions_df['tags'] = versions_df['tags'].apply(
                lambda x: x['tags'])
        except KeyError as e:
            logging.error(e)

        return versions_df

    def search_experiments(self):
        """
        Search and retrieve a list of MLflow experiments.

        Returns:
            pandas.DataFrame: A DataFrame containing experiment information.
        """
        experiments = mlflow.search_experiments()
        return experiments

    def register_model(self, run_id, model_name, tags=None):
        """
        Register a model with MLflow.

        Args:
            model_uri (str): The URI of the model.
            model_name (str): The name to register the model with.
            tags (dict): Tags to associate with the model.

        Returns:
            None
        """
        run_uri = self.get_model_uri(run_id, prediction=True)
        run_uri = run_uri.replace("mlflow-artifacts:", "runs:")
        # Register the model
        mlflow.register_model(run_uri, model_name, tags=tags)

    def create_model_version(self, run_id, model_name, tags=None):
        """
        Create a new version of a model.

        Args:
            run_id (str): The run ID of the model to create a version of.
            model_name (str): The name of the model to create a version of.
            tags (dict): Tags to associate with the model.

        Returns:
            None
        """
        run_uri = self.get_model_uri(run_id, prediction=True)
        self.client.create_model_version(
            name=model_name, source=run_uri, run_id=run_id, tags=tags
        )

    def transition_model_version_stage(
        self, name, version, stage, archive_existing_versions=True
    ):
        """
        Transition a model version to a new stage.

        Args:
            name (str): The name of the model.
            version (str): The version of the model.
            stage (str): The new stage to transition to.
            archive_existing_versions (bool): Whether to archive existing versions.

        Returns:
            None
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def delete_registered_model(self, model_name):
        """
        Delete a registered model.

        Args:
            model_name (str): The name of the model to delete.

        Returns:
            None
        """
        self.client.delete_registered_model(model_name)
    
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

    def list_runs(self, experiment_id:str) -> None | pd.DataFrame:
        """
        List runs for a specified MLflow experiment.

        Args:
            experiment_id (str): The experiment_id to retrieve runs from.

        Returns:
            pandas.DataFrame: A DataFrame containing run information.
        """
        try:
            # Tenta buscar as runs do experimento
            runs = mlflow.search_runs(experiment_ids=[experiment_id])
        except SientiaMlException as e:
            print(f"\nOcorreu um erro ao listar as runs: {e}\n")
            return None
            
        # Verifica se a coluna 'tags.mlflow.runName' está presente no DataFrame
        if 'tags.mlflow.runName' in runs.columns:
            run_names = runs['tags.mlflow.runName']
            runs.drop(columns=['tags.mlflow.runName'], inplace=True)
            runs.insert(0, 'run_name', run_names)
        else:
            # Se a coluna não existir, define 'run_name' com valor nulo ou outro padrão
            runs.insert(0, 'run_name', None)
            print("\nAviso: 'tags.mlflow.runName' não está presente nas runs.\n")

        # Lista de colunas a serem removidas, verificando se elas existem
        columns_to_drop = [
            'experiment_id', 'status', 'artifact_uri', 'tags.mlflow.source.type',
            'tags.mlflow.source.name', 'tags.mlflow.user', 'tags.mlflow.log-model.history'
        ]
        # Filtra as colunas para remover apenas as que estão no DataFrame
        columns_to_drop = [col for col in columns_to_drop if col in runs.columns]
        runs.drop(columns=columns_to_drop, inplace=True)

        return runs

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
                latest_production_id = self.get_model_run_id(model_name)
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
            latest_production_id = self.get_model_run_id(
                model_name=model_id, stage="Production"
            )
            # print('cheguei aqui, Model', latest_production_id)
            model_uri = self.get_model_uri(latest_production_id, prediction=False)
        elif by == "run":
            model_uri = self.get_model_uri(run_id=model_id, prediction=False)
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

    def get_next_run_name(self, project_name):
        runs = mlflow.search_runs(
            experiment_names=[project_name], order_by=["start_time desc"]
        )
        next_run_number = len(runs) + 1
        return f"{project_name}-{next_run_number}"

    def retrain_model(self, data: pd.DataFrame, model_name: str):
        """
        Retrain a model with new data.

        Args:
            data (pandas.DataFrame): The new data to use for retraining.
            model_name (str): The name of the model to retrain.

        Returns:
            mlflow.sklearn.Model: The retrained prediction model.
            mlflow.sklearn.Model: The retrained data model.
            mse (float): The mean squared error of the retrained model.
            r2 (float): The R-squared score of the retrained model.
        """
        # Get the model URI
        latest_production_id = self.get_model_run_id(model_name)
        pred_model_uri = self.get_model_uri(latest_production_id, prediction=True)
        data_model_uri = self.get_model_uri(latest_production_id, prediction=False)

        data_model = mlflow.sklearn.load_model(data_model_uri)
        prediction_model = mlflow.sklearn.load_model(pred_model_uri)

        data_model.fit(data)
        df_for_pred = data_model.get_treated_data()
        X_train, X_test, y_train, y_test = train_test_split(
            data[data_model.variable_columns],
            data[data_model.target_variable],
            train_size=0.8,
        )
        # Retrain the model
        prediction_model = prediction_model.fit(X_train, y_train)
        # Setting run name

        y_pred = prediction_model.predict(X_test)
        mse = round(
            mean_squared_error(y_test.astype(np.float64), y_pred.astype(np.float64)), 2
        )
        r2 = round(r2_score(y_test.astype(np.float64), y_pred.astype(np.float64)), 2)

        reference_data = pd.concat([X_train, y_train], axis=1)

        reference_data.rename(
            columns={data_model.target_variable: "target"}, inplace=True
        )
        reference_data["prediction"] = prediction_model.predict(X_train)

        current_data = pd.concat([X_test, y_test], axis=1)
        current_data.rename(
            columns={data_model.target_variable: "target"}, inplace=True
        )
        current_data["prediction"] = y_pred

        report = Reports(
            reference_data=reference_data.astype(np.float64),
            current_data=current_data.astype(np.float64),
            base_path="./reports/retrain",
        )

        report.add_data_quality_section(
            columns=data_model.variable_columns + ["target"]
        )
        report.add_data_drift_section(columns=data_model.variable_columns + ["target"])
        report.add_regression_section()

        report_path = "reports/retrain/report.html"
        report.save_all_sections_html(report_path)

        data_path = "./data/retrain/train_data.csv"
        reference_data.to_csv(data_path, index=False)

        def get_next_run_name(project_name):
            runs = mlflow.search_runs(
                experiment_names=[model_name], order_by=["start_time desc"]
            )
            next_run_number = len(runs) + 1
            return f"{project_name}-{next_run_number}"

        current_run_name = get_next_run_name(model_name)

        experiment_description = "Retrain model due to drift detected"
        mlflow.set_experiment(model_name)
        with mlflow.start_run(
            run_name=current_run_name, description=experiment_description
        ) as run:
            mlflow.log_params({"model_type": "Linear Regression"})
            mlflow.log_params({"lag": data_model.lag})
            mlflow.log_params({"ma": data_model.window})
            mlflow.log_params({"Retrain": True})
            mlflow.log_params({"low_lim": data_model.low_lim})
            mlflow.log_params({"normalized": data_model.use_scaler})
            mlflow.log_params({"ar": data_model.include_ar})
            mlflow.log_params({"Removed_intervals": None})
            mlflow.log_metrics({"MSE": mse, "R2": r2})
            mlflow.sklearn.log_model(data_model, "data_model")
            mlflow.sklearn.log_model(
                prediction_model,
                "prediction_model",
                extra_pip_requirements=[
                    "git+https://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@github.com/Aignosi/sientia-mlops-library.git"
                ],
            )

            mlflow.log_artifact(report_path)
            mlflow.log_artifact(data_path)

        return run.info.run_id, mse, r2

    def log_prediction_model(self,sk_model: Any) -> None:
        """
            Log a sklearn model.

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
            Log a sklearn model.

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
        Log a sklearn model.

        Args:
            sk_model: scikit-learn model to be saved.
            artifact_path: Run-relative artifact path.

        Returns:
            None
        """
        mlflow.sklearn.log_model(
            sk_model, 
            artifact_path, 
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

    def set_tag(self,key:str, value:Any) -> None:
        """
        Set a tag under the current run.

        Args:
            key (str): tag name
            value (any): tag value

        Returns:
            None
        """
        mlflow.set_tag(key,value)

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

    def get_model_experiment_id(self, model_name: str):
        """
        Get the project name associated with a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: The project name.
        """
        latest_production_id = self.get_model_run_id(
            model_name=model_name, stage="Production"
        )
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

    def get_run_id(self, run_name: str):
        """
        Get the run ID associated with a run name.

        Args:
            run_name (str): The run name.

        Returns:
            str: The run ID.
        """
        if run_name:
            # run_id is in the first row of  of the DataFrame
            run_ID = mlflow.search_runs(
                filter_string=f'tags.mlflow.runName = "{run_name}"',
                search_all_experiments=True,
            ).iloc[0]["run_id"]
        else:
            run_ID = None
        return run_ID

    def get_model_metrics(
        self,
        reference_data: pd.DataFrame,
        real_data: pd.DataFrame,
        predictions: pd.Series,
        type_flag: str,
    ) -> dict:
        """
        Get metrics for a model.

        Args:
            reference_data (pd.DataFrame): The reference data. None if drift detection is not used.
            real_data (pd.DataFrame): The real data.
            predictions (pd.Series): The predictions.
            type_flag (str): The type of model to get metrics for. Ex: 'regression', 'classification'.

        Returns:
            dict: A dictionary containing the metrics: mse and r2.
        """
        metrics = {"data_drift": {}, "data_quality": {}, "regression": {}}

        if type_flag == "regression":
            # Calculando as métricas
            error = real_data["target"] - predictions
            abs_error = np.abs(error)
            abs_error_max = np.max(abs_error)
            mean_abs_error = np.mean(abs_error)
            mean_abs_perc_error = np.mean(np.abs((error) / real_data["target"])) * 100
            mean_error = np.mean(error)
            r2_scr = r2(real_data["target"], predictions)
            rmse = np.sqrt(mse(real_data["target"], predictions))
            mean_squared_error = mse(real_data["target"], predictions)

            # Adicionando as novas métricas ao dicionário existente
            metrics["regression"]["abs_error_max"] = abs_error_max
            metrics["regression"]["mean_abs_error"] = mean_abs_error
            metrics["regression"]["mean_abs_perc_error"] = mean_abs_perc_error
            metrics["regression"]["mean_error"] = mean_error
            metrics["regression"]["r2_score"] = r2_scr
            metrics["regression"]["rmse"] = rmse
            metrics["regression"]["mse"] = mean_squared_error

            if reference_data is not None:
                target_drift = rce_drift(reference_data, real_data, "target")
                prediction_drift = rce_drift(reference_data, real_data, "prediction")

                model_drift = abs(target_drift - prediction_drift)
                # model_drift = (model_drift - min(model_drift)) / (max(model_drift) - min(model_drift))

                metrics["data_drift"]["target_drift"] = target_drift.mean()
                metrics["data_drift"]["prediction_drift"] = prediction_drift.mean()
                metrics["data_drift"]["model_drift"] = model_drift.mean()

        elif type_flag == "classification":
            metrics = {}
        return metrics
    
    def get_target_variable_name(self, model_name: str) -> str:
        """
        Fetches the target_variable column from the latest production model's run in MLflow.

        Parameters:
            model_name (str): Name of the model in the MLflow Model Registry.

        Returns:
            str: The target variable column from the logged dataset.
        """
        run_id = self.get_model_run_id(model_name)
        run = self.client.get_run(run_id)

        # Tenta obter a variável alvo sem capturar exceções
        target_series = run.data.params.get("target_variable") or run.data.params.get("Target")

        if not target_series:
            raise ValueError("No value found for 'target_variable' or 'Target'.")

        if isinstance(target_series, (list, tuple)) and target_series:
            return pd.Series(target_series).values[0]
        elif isinstance(target_series, str):
            return target_series
        else:
            raise TypeError(f"Unexpected type for target_series: {type(target_series)}")

def initialize_tracker(
        config_file: str|None = None, 
        url: str|None = None, 
        username: str|None = None, 
        password: str|None = None) -> ModelServing:
    '''
    Initialize the SIENTIA™ Tracker a.k.a ModelServing class
    Args:
        config_file: path of the .yaml file that contains the information about the configuration of the tracker. Tipically, this may be 'config.yaml'. The file must specify the following fields:
          * tracker_url : the url of the tracker server
          * tracker_user : client username 
          * tracker_password: client password
    
    Returns:
        ModelServing: Tracker object, instance of ModelServing class
    '''
    if config_file is not None:
        config = yaml.safe_load(open(config_file))
        MLFLOW_TRACKING_URL = os.getenv(
            'MLFLOW_TRACKING_URL', config['tracker_url'])
        MLFLOW_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME', config['tracker_user'])
        MLFLOW_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD', config['tracker_password'])
    else:
        assert url is not None, "URL undefined"
        assert username is not None, "username undefined"
        assert password is not None, "password undefined"
        MLFLOW_TRACKING_URL = url
        MLFLOW_USERNAME = username
        MLFLOW_PASSWORD = password
    
    # Initialize the SIENTIA™ Tracker with the server URL
    sientia_tracker = ModelServing(
        MLFLOW_TRACKING_URL, MLFLOW_USERNAME, MLFLOW_PASSWORD)
    
    return sientia_tracker
