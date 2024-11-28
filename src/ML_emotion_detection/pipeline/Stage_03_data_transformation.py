"""
This module contains the DataTransformationTrainingPipeline class, which is
responsible for managing the data transformation process in the pipeline.
"""

from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.data_transformation import \
    DataTransformation
from pathlib import Path


STAGE_NAME = "Data Transformation stage"


class DataTransformationTrainingPipeline:
    """
    A class to manage the data transformation process, including
    configuration setup, data validation status check, and applying
    transformations to training and test data.
    """

    def __init__(self):
        """
        Initialize the DataTransformationTrainingPipeline class.

        The constructor is empty, as the configuration and transformation steps
        are handled in the `main` method.
        """
        pass

    def main(self):
        """
        Main method to execute the data transformation pipeline.

        This method checks the data validation status, retrieves the data
        transformation configuration, and applies the necessary transformations
        to the training and test data.

        Returns:
            None

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            # Read the status of data validation from a file
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            # Proceed only if data validation was successful
            if status == "True":
                # Retrieve configuration for data transformation
                config = ConfigurationManager()
                data_transformation_config = config.\
                    get_data_transformation_config()

                # Initialize the DataTransformation class with the
                # retrieved config
                data_transformation = DataTransformation(
                    config=data_transformation_config
                )

                # Split the data into training and test datasets
                Tr_X, Te_X = data_transformation.train_test_spliting()

                # Apply the data transformation pipe to the train and test data
                data_transformation.pipeline_and_transform(Tr_X, Te_X)

        except Exception as e:
            # Raise the exception if an error occurs
            raise e
