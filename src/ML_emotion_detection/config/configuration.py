"""
This module contains the ConfigurationManager class for managing
configuration settings in the ML Emotion Detection project.
"""

from src.ML_emotion_detection.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from src.ML_emotion_detection.utils.common import read_yaml, create_directories
from src.ML_emotion_detection import logger
from src.ML_emotion_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    """
    A class to handle configuration settings for various stages
    of the ML pipeline, including data ingestion, validation,
    transformation, training, and evaluation.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        """
        Initialize ConfigurationManager with paths to config, params,
        and schema files.

        Args:
            config_filepath (str): Path to the configuration YAML file.
            params_filepath (str): Path to the parameters YAML file.
            schema_filepath (str): Path to the schema YAML file.
        """
        logger.info("Reading CONFIG YAML FROM %s", config_filepath)
        self.config = read_yaml(config_filepath)

        logger.info("Reading PARAMS YAML FROM %s", params_filepath)
        self.params = read_yaml(params_filepath)

        logger.info("Reading SCHEMA YAML FROM %s", schema_filepath)
        self.schema = read_yaml(schema_filepath)

        logger.info(
            "Creating directory stored at artifacts_root \
                    of CONFIG YAML"
        )
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieve and configure the Data Ingestion settings.

        Returns:
            DataIngestionConfig: Configuration settings for data ingestion.
        """
        config = self.config.data_ingestion

        logger.info("Creating directory stored at root_dir of CONFIG YAML")
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Retrieve and configure the Data Validation settings.

        Returns:
            DataValidationConfig: Configuration settings for data validation.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Retrieve and configure the Data Transformation settings.

        Returns:
            DataTransformationConfig: Configuration settings for data
            transformation.
        """
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieve and configure the Model Trainer settings.

        Returns:
            ModelTrainerConfig: Configuration settings for model training.
        """
        config = self.config.model_trainer
        params = self.params.LogisticRegression

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path_x=config.train_data_path_x,
            train_data_path_target=config.train_data_path_target,
            test_data_path_x=config.test_data_path_x,
            test_data_path_target=config.test_data_path_target,
            model_name=config.model_name,
            class_weight=params.class_weight,
            max_iter=params.max_iter,
            penalty=params.penalty,
            solver=params.solver,
            n_jobs=params.n_jobs,
            random_state=params.random_state,
        )
        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieve and configure the Model Evaluation settings.

        Returns:
            ModelEvaluationConfig: Configuration settings for model evaluation.
        """
        config = self.config.model_evaluation
        params = self.params.LogisticRegression

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path_x=config.test_data_path_x,
            test_data_path_target=config.test_data_path_target,
            model_path=config.model_path,
            all_params=params,
            metric_file_name=config.metric_file_name,
            cm_file_name=config.cm_file_name,
            mlflow_uri="https://dagshub.com/ArghyadipB/ \
            Emotion-Detection-using-ML.mlflow",
        )
        return model_evaluation_config
