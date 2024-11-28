"""
This module contains the DataValidationTrainingPipeline class, which is
responsible for managing the data validation process in the pipeline.
"""

from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.data_validation import DataValidation

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    """
    A class to manage the data validation process, including
    configuration setup and column validation.
    """

    def __init__(self):
        """
        Initialize the DataValidationTrainingPipeline class.

        The constructor is empty, as the configuration and validation steps
        are handled in the `main` method.
        """
        pass

    def main(self):
        """
        Main method to execute the data validation pipeline.

        This method retrieves the data validation configuration, initializes
        the `DataValiadtion` class with the configuration, and validates
        the columns of the dataset.

        Returns:
            None
        """
        # Retrieve configuration for data validation
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()

        # Initialize the DataValiadtion class with the retrieved config
        data_validation = DataValidation(config=data_validation_config)

        # Validate the columns in the dataset
        data_validation.validate_all_columns()
