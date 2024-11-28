"""
This module contains the DataIngestion class for handling data
downloading and extraction as part of the data ingestion pipeline.
"""

import os
import urllib.request as request
import zipfile
from src.ML_emotion_detection.utils.common import get_size
from src.ML_emotion_detection.entity.config_entity import DataIngestionConfig
from src.ML_emotion_detection import logger
from pathlib import Path


class DataIngestion:
    """
    A class to manage the ingestion of data, including downloading
    and extracting files.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion class.

        Args:
            config (DataIngestionConfig): Configuration object containing
                file paths and source URLs for data ingestion.
        """
        self.config = config

    def download_file(self):
        """
        Download the data file from the source URL if it does not already
        exist.

        Checks if the local data file exists. If not, downloads the file
        from the specified source URL and logs the download information.
        If the file exists, logs its size.

        Returns:
            None
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading file from {self.config.source_url}.")
            filename, headers = request.urlretrieve(
                url=self.config.source_url, filename=self.config.local_data_file
            )
            logger.info(
                f"{filename} downloaded successfully \
                        with the following info:\n{headers}"
            )
        else:
            file_size = get_size(Path(self.config.local_data_file))
            logger.info(f"File already exists with size: {file_size}")

    def extract_zip_file(self):
        """
        Extract the downloaded zip file to the specified directory.

        Ensures that the extraction directory exists and extracts the contents
        of the zip file into it.

        Returns:
            None
        """
        try:
            logger.info(
                f"Extracting file: {self.config.local_data_file} \
                        to {self.config.unzip_dir}."
            )
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extraction completed successfully at {unzip_path}.")
        except zipfile.BadZipFile as e:
            logger.error(
                f"Failed to extract {self.config.local_data_file}: \
                         {e}"
            )
            raise e
