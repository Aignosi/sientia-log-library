import logging
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sientia.reports import Reports
from sientia.metrics import r2, mse, rce_drift
from sientia.exceptions import SientiaMlException
import pandas as pd
import numpy as np
import mlflow
from run_tracker import RunTracker

class ModelTracker:
    def __init__(self, client):
        self.client: mlflow.tracking.MlflowClient = client
        self.run_tracker = RunTracker
    
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
        
    def get_models(self):
        """
        Get information about registered models.

        Returns:
            pandas.DataFrame: A DataFrame containing model information, such as:

                - name (str): Name of the registered model.
                - creation_timestamp (int): When the model was created (in milliseconds since epoch).
                - last_updated_timestamp (int): When the model was last updated (in milliseconds since epoch).
                - description (str): Optional description of the model.
                - latest_versions (List[ModelVersion]): List of the latest versions per stage (e.g., Production, Staging).
                - tags (List[RegisteredModelTag]): Custom tags attached to the model.
                - aliases (Dict[str, str]), optional): Alias names pointing to specific model versions (e.g., {"prod": "5"}).
                - id (str, optional): Unique internal identifier of the model (depending on MLflow backend).

        """

        registered_models = mlflow.search_registered_models()
        model_data = []

        for rm in registered_models:
            model_info = dict(rm)
            model_data.append(model_info)

        df_model_info = pd.DataFrame(model_data)
        return df_model_info

    # Function to get information about model versions
    def models_info(self, model_name: list) -> pd.DataFrame:
        """
        Get information about model versions for a given model name.

        Args:
            model_name (list): The name of the model.

        Returns:
            pandas.DataFrame: A DataFrame containing the version, current stage, run id,
            associated tags and name of the model. 
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

        current_run_name = self.run_tracker.get_next_run_name(model_name)

        experiment_description = "Retrain model due to drift detected"
        mlflow.set_experiment(model_name)
        with mlflow.start_run(
            run_name=current_run_name, description=experiment_description
        ) as run:
            mlflow.log_params({"model_name": "Linear Regression"})
            mlflow.log_params({"lag_train": data_model.lag_train})
            mlflow.log_params({"lag_val": data_model.lag_val})
            mlflow.log_params({"retrain": True})
            mlflow.log_params({"lower_limits": data_model.lower_limits})
            mlflow.log_params({"upper_limits": data_model.upper_limits})
            mlflow.log_params({"scaler_name": data_model.scaler_name})
            mlflow.log_params({"scaler_params": data_model.scaler_params})
            mlflow.log_params({"created_lags": data_model.created_lags})
            mlflow.log_params({"removed_intervals": None})
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