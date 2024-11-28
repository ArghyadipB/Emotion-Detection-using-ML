"""
This module contains configuration classes for various stages of the ML
pipeline:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

Each class contains configuration parameters relevant to their respective
stages.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion stage.
    Contains paths for root directory, source URL, and the location of
    the local and unzipped data.
    """

    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the Data Validation stage.
    Contains paths for root directory, status file, and schema definitions.
    """

    root_dir: Path
    status_file: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration for the Data Transformation stage.
    Contains paths for root directory and transformed data.
    """

    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the Model Training stage.
    Contains paths for training and test data, model parameters,
    and solver settings.
    """

    root_dir: Path
    train_data_path_x: Path
    train_data_path_target: Path
    test_data_path_x: Path
    test_data_path_target: Path
    model_name: str

    # model parameters
    class_weight: str
    max_iter: int
    penalty: str
    solver: str
    n_jobs: int
    random_state: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration for the Model Evaluation stage.
    Contains paths for test data, model, and evaluation metrics.
    """

    root_dir: Path
    test_data_path_x: Path
    test_data_path_target: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    cm_file_name: Path
    mlflow_uri: str
