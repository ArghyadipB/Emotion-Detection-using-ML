from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()