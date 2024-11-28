from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from src.ML_emotion_detection.utils.common import load_bin, save_json
import mlflow
import dagshub
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from pathlib import Path
from src.ML_emotion_detection.config.configuration import ModelEvaluationConfig


class ModelEvaluation:
    """
    A class to evaluate machine learning models and log results into MLflow.

    Attributes:
        config (ModelEvaluationConfig): Configuration object containing file
        paths, parameters, and other settings.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes the ModelEvaluation instance with the provided
        configuration.

        Args:
            config (ModelEvaluationConfig): Configuration object containing
            paths for test data, model files, and output directories.
        """
        self.config = config

    def eval_metrics(self, actual, pred):
        """
        Evaluates classification metrics and generates a confusion matrix.

        Args:
            actual (list[int]): True labels of the test data.
            pred (list[int]): Predicted labels from the model.

        Returns:
            tuple: A tuple containing the following metrics:
                - accuracy (float): Accuracy of the predictions.
                - weighted_precision (float): Weighted precision score.
                - weighted_recall (float): Weighted recall score.
                - weighted_f1 (float): Weighted F1 score.
                - cm_disp (ConfusionMatrixDisplay): Display object for the
                confusion matrix.
        """
        accuracy = accuracy_score(actual, pred)
        weighted_precision = precision_score(actual, pred, average="weighted")
        weighted_recall = recall_score(actual, pred, average="weighted")
        weighted_f1 = f1_score(actual, pred, average="weighted")

        label_map = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }

        y_true_mapped = [label_map[label] for label in actual]
        y_pred_mapped = [label_map[label] for label in pred]

        cm = confusion_matrix(y_true_mapped, y_pred_mapped)
        cm_disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(label_map.values())
        )

        return accuracy, weighted_precision, weighted_recall, \
            weighted_f1, cm_disp

    def log_into_mlflow(self):
        """
        Logs model evaluation metrics, artifacts, and model information into
        MLflow.

        This method performs the following steps:
        1. Loads the test data and model.
        2. Predicts outcomes using the loaded model.
        3. Evaluates the predictions using evaluation metrics.
        4. Saves and logs the confusion matrix as an artifact.
        5. Logs evaluation metrics into MLflow.
        6. Registers the model with MLflow.

        Raises:
            FileNotFoundError: If any specified file paths in the
            configuration are invalid.
        """
        test_data_X = load_bin(self.config.test_data_path_x)
        test_data_target = load_bin(self.config.test_data_path_target)
        model = load_bin(self.config.model_path)

        dagshub.init(
            repo_owner="ArghyadipB", repo_name="Emotion-Detection-using-ML",
            mlflow=True
        )
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_output = model.predict(test_data_X)
            (acc, weighted_pre, weighted_rec, weighted_f1, conf_mat) = (
                self.eval_metrics(test_data_target, predicted_output)
            )

            conf_mat.plot(cmap=plt.cm.Blues, values_format="d")
            plt.savefig(self.config.cm_file_name)
            plt.close()

            scores = {
                "Accuracy": acc,
                "Weighted_precision": weighted_pre,
                "Weighted_recall": weighted_rec,
                "Weighted_F1": weighted_f1,
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Weighted Precision", weighted_pre)
            mlflow.log_metric("Weighted Recall", weighted_rec)
            mlflow.log_metric("Weighted F1", weighted_f1)
            mlflow.log_artifact(self.config.cm_file_name)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="LogisticRegression"
                )
            else:
                mlflow.sklearn.log_model(model, "model")
