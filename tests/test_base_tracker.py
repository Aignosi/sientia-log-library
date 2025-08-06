import os
from unittest.mock import MagicMock, patch

import pytest
from sientia_tracker.base_tracker import BaseTracker


@pytest.fixture
def mock_mlflow_client():
    with patch("mlflow.tracking.MlflowClient") as mock_client:
        yield mock_client


@pytest.fixture
def mock_set_tracking_uri():
    with patch("mlflow.set_tracking_uri") as mock_set_uri:
        yield mock_set_uri


def test_init_valid_uri_no_auth(mock_mlflow_client, mock_set_tracking_uri):
    tracking_uri = "http://sientia-platform.com"
    tracker = BaseTracker(tracking_uri)
    mock_set_tracking_uri.assert_called_with(tracking_uri)
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == ""
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == ""
    assert isinstance(tracker.client, MagicMock)


def test_init_valid_uri_with_auth(mock_mlflow_client, mock_set_tracking_uri):
    tracking_uri = "http://sientia-platform.com"
    username = "user"
    password = "pass"
    tracker = BaseTracker(tracking_uri, username, password)
    mock_set_tracking_uri.assert_called_with(tracking_uri)
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == username
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == password
    assert isinstance(tracker.client, MagicMock)


def test_init_invalid_uri(mock_mlflow_client, mock_set_tracking_uri):
    tracking_uri = " invalid-uri"
    # Simular o comportamento real: não lança exceção, apenas configura a URI
    BaseTracker(tracking_uri)
    mock_set_tracking_uri.assert_called_with(tracking_uri)
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == ""
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == ""


def test_init_empty_uri(mock_mlflow_client, mock_set_tracking_uri):
    tracking_uri = ""
    # Simular o comportamento real: não lança exceção, apenas configura a URI
    BaseTracker(tracking_uri)
    mock_set_tracking_uri.assert_called_with(tracking_uri)
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == ""
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == ""


def test_init_none_uri(mock_mlflow_client, mock_set_tracking_uri):
    tracking_uri = None
    # Simular o comportamento real: não lança exceção, apenas configura a URI
    BaseTracker(tracking_uri)
    mock_set_tracking_uri.assert_called_with(tracking_uri)
    assert os.environ["MLFLOW_TRACKING_USERNAME"] == ""
    assert os.environ["MLFLOW_TRACKING_PASSWORD"] == ""
