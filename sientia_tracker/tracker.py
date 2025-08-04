
import os
import yaml
from sientia_tracker.model_serving.model_serving import ModelServing
# remove warnings
import warnings

warnings.filterwarnings("ignore")

import os

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
