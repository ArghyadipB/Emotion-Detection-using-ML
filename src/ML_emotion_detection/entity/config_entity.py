# goes to src/entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict
    

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path_X: Path
    train_data_path_target: Path
    test_data_path_X: Path
    test_data_path_target: Path
    model_name: str
    
    # model
    class_weight: str
    max_iter: int
    penalty: str
    solver: str
    n_jobs: int
    random_state: int
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path_X: Path
    test_data_path_target: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
     