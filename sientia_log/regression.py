
import mlflow
from typing import Any, List
from basic import SimpleLog

class RegressionLog(SimpleLog):
    """
    Log object for regression models.
    """
    def __init__(self,model: Any,
                    model_name: str = "regr_model",
                    target: str = 'target',
                    inputs: str | List[str] ='input',
                    dataset_name: str = "data",
                    date_column: str = "date",
                    r2: float = 1.0, 
                    train_size: float = 0.8,
                    shuffle: bool = True,
                ) -> None:
        """
        params:
            model: Scikit-Learn Regression Model
            model_name: The name of the model as defined by user. Default: regr_model
            target: The name of the target. Default: target
            inputs: The name of the inputs. Default: inputs
            dataset_name: The name of the dataset. Default: data
            date_column: The name of the date column. Default: date
            r2: r2-score of the model. Default: 1
            train_size: Proportion of the data used to traind the model. Default: 0.8
            shuffle: Wheter or not the train data was shuffled. Default: True
        """
        super().__init__(self,dataset_name,inputs)

        mlflow.log_params({
            "Model": "Linear Regression",
            "Date Column": date_column,
            "Target": target,
            "Train Size": train_size,
            "Shuffle": shuffle,
            })
        
        mlflow.log_model(model,model_name)
        mlflow.log_metrics({"r2":r2})
