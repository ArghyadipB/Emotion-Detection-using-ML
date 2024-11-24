from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from src.ML_emotion_detection.utils.common import load_bin, save_json
from pathlib import Path
from src.ML_emotion_detection.config.configuration import ModelEvaluationConfig


class ModelEvaluation:
    """
    A class to evaluate a machine learning model's performance based on metrics such as 
    accuracy, precision, recall, and F1 score. The class provides methods to evaluate 
    the model's performance and save the results to a specified location.

    Attributes:
        config (ModelEvaluationConfig): Configuration object containing paths and settings 
                                         needed for model evaluation.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation object with the provided configuration.

        Args:
            config (ModelEvaluationConfig): The configuration object containing paths to 
                                             test data and model, and the filename for saving 
                                             evaluation metrics.
        """
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Evaluates the model performance using various metrics: accuracy, weighted precision, 
        weighted recall, and weighted F1 score.

        Args:
            actual (array-like): The true labels of the test dataset.
            pred (array-like): The predicted labels from the model.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): The accuracy of the model.
                - weighted_precision (float): The weighted precision score of the model.
                - weighted_recall (float): The weighted recall score of the model.
                - weighted_f1 (float): The weighted F1 score of the model.
        """
        accuracy = accuracy_score(actual, pred)
        weighted_precision = precision_score(actual, pred, average='weighted')
        weighted_recall = recall_score(actual, pred, average='weighted')
        weighted_f1 = f1_score(actual, pred, average='weighted')
        return accuracy, weighted_precision, weighted_recall, weighted_f1

    def save_results(self):
        """
        Loads test data and the trained model from the specified paths, makes predictions, 
        evaluates the performance, and saves the results to a JSON file.

        The results include accuracy, weighted precision, weighted recall, and weighted F1 
        score. The evaluation scores are saved to a JSON file specified in the configuration.

        This method assumes the following files exist:
            - The test feature data (X) is stored in a binary format.
            - The test target data (y) is stored in a binary format.
            - The trained model is stored in a binary format.
            - The metric results are saved as a JSON file.

        Raises:
            FileNotFoundError: If any of the required files are not found at the specified 
                                paths.
        """
        test_data_X = load_bin(self.config.test_data_path_X)
        test_data_target = load_bin(self.config.test_data_path_target)
        model = load_bin(self.config.model_path)
        
        predicted_output = model.predict(test_data_X)
        (acc, weighted_pre, weighted_rec, weighted_f1) = self.eval_metrics(test_data_target,
                                                                          predicted_output)

        scores = {"Accuracy": acc, "Weighted_precision": weighted_pre,
                  "Weighted_recall": weighted_rec, "Weighted_F1": weighted_f1}
        
        save_json(path=Path(self.config.metric_file_name), data=scores)
       