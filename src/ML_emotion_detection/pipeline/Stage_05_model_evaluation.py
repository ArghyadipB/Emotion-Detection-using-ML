"""
This module contains the ModelEvaluationPipeline class, which is
responsible for performing model evaluation and logging results into MLflow.
"""

from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model evaluation stage"


class ModelEvaluationPipeline:
    """
    A class to manage the model evaluation process, including
    configuration setup and logging results into MLflow.
    """

    def __init__(self):
        """
        Initialize the ModelEvaluationPipeline class.

        The constructor is empty, as the configuration and evaluation steps
        are handled in the `main` method.
        """
        pass

    def main(self):
        """
        Main method to execute the model evaluation pipeline.

        This method retrieves the model evaluation configuration, initializes
        the `ModelEvaluation` class with the configuration, and logs the
        results into MLflow.

        Returns:
            None
        """
        # Retrieve configuration for model evaluation
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()

        # Initialize the ModelEvaluation class with the retrieved config
        model_evaluation = ModelEvaluation(config=model_evaluation_config)

        # Log evaluation results into MLflow
        model_evaluation.log_into_mlflow()
