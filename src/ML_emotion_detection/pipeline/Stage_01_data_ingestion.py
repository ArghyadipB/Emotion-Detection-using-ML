"""
This module contains the DataIngestionTrainingPipeline class, which is
responsible for managing the data ingestion process in the pipeline.
"""

from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.data_ingestion import DataIngestion

STAGE_NAME = "DATA INGESTION"


class DataIngestionTrainingPipeline:
    """
    A class to manage the data ingestion process, including
    configuration setup, file download, and extraction.
    """

    def __init__(self):
        """
        Initialize the DataIngestionTrainingPipeline class.

        The constructor is empty, as the configuration and ingestion steps
        are handled in the `main` method.
        """
        pass

    def main(self):
        """
        Main method to execute the data ingestion pipeline.

        This method retrieves the data ingestion configuration, initializes
        the `DataIngestion` class with the configuration, and executes the
        file download and extraction processes.

        Returns:
            None
        """
        # Retrieve configuration for data ingestion
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        # Initialize the DataIngestion class with the retrieved config
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Download the file from the source URL
        data_ingestion.download_file()

        # Extract the downloaded zip file
        data_ingestion.extract_zip_file()
