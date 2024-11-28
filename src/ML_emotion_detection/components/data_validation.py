"""
This module contains the DataValidation class for validating
data columns based on a predefined schema.
"""

from src.ML_emotion_detection.entity.config_entity import DataValidationConfig
from src.ML_emotion_detection import logger
import pandas as pd


class DataValidation:
    """
    A class to perform data validation tasks for ensuring the integrity of
    data used in the ML pipeline.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initialize the DataValidation class.

        Args:
            config (DataValidationConfig): Configuration object containing
                paths and schema information for validation.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        Validate whether all required columns are present in the dataset.

        Reads the dataset from the specified directory and checks if all
        columns defined in the schema are present.

        Returns:
            bool: True if all required columns are present, False otherwise.

        Raises:
            Exception: If an error occurs during the validation process.
        """
        try:
            validation_status = None

            logger.info("Reading data from the specified directory.")
            data = pd.read_parquet(self.config.unzip_data_dir,
                                   engine="pyarrow")
            all_cols = list(data.columns)

            logger.info("Fetching the required schema columns.")
            all_schema = self.config.all_schema.keys()

            logger.info("Checking if the required columns exist or not.")
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    logger.warning(f"Column missing: {col}")
                    with open(self.config.status_file, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                    break  # Break early if a column is missing
                else:
                    validation_status = True

            with open(self.config.status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            logger.error(f"An error occurred during validation: {e}")
            raise e
