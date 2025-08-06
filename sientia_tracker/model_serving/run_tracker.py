import logging
from sientia.exceptions import SientiaMlException
from typing import Any
import pandas as pd
import mlflow


class RunTracker:
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
    
    def get_next_run_name(self, project_name: str) -> str:
        '''
        Generates a name for a new run based on the current number of runs in a 
        project.

        Args:
            project_name (str): Name of the project in which the run will be saved

        Returns:
            str: Next run name
        '''
        runs = self.search_runs_by_name(
            experiment_names=[
                project_name
            ], order_by=["start_time desc"])
        if len(runs) == 0:
            return f"{project_name}-1"
        run_names = runs['tags.mlflow.runName'].to_list()
        
        # Extract run numbers from existing run names
        run_numbers = []
        prefix = f"{project_name}-"
        for run_name in run_names:
            if run_name.startswith(prefix):
                try:
                    # Extract the number part after the prefix
                    num = int(run_name[len(prefix):])
                    run_numbers.append(num)
                except (ValueError, IndexError):
                    continue
        
        # Get the next run number (1 if no runs found, max + 1 otherwise)
        next_run_number = max(run_numbers) + 1 if run_numbers else 1
        return f"{project_name}-{next_run_number}"
    
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
