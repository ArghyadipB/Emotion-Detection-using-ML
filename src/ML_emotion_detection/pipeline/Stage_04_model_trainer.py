"""
This module contains the ModelTrainerTrainingPipeline class, which is
responsible for managing the model training process.
"""

from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.model_trainer import ModelTrainer


STAGE_NAME = "Model Trainer stage"


class ModelTrainerTrainingPipeline:
    """
    A class to manage the model training process, including
    configuration setup and executing the training.
    """

    def __init__(self):
        """
        Initialize the ModelTrainerTrainingPipeline class.

        The constructor is empty, as the configuration and training steps
        are handled in the `main` method.
        """
        pass

    def main(self):
        """
        Main method to execute the model training pipeline.

        This method retrieves the model trainer configuration, initializes
        the `ModelTrainer` class with the configuration, and runs the
        training process.

        Returns:
            None
        """
        # Retrieve configuration for model training
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Initialize the ModelTrainer class with the retrieved config
        model_trainer = ModelTrainer(config=model_trainer_config)

        # Start the model training process
        model_trainer.train()
