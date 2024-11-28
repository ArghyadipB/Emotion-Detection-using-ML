"""
This script runs the various stages of the ML emotion detection pipeline:
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

Each stage is encapsulated in a separate pipeline class, and the execution of
 each stage is logged.
"""

from src.ML_emotion_detection import logger
from src.ML_emotion_detection.pipeline.Stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.ML_emotion_detection.pipeline.Stage_02_data_validation import (
    DataValidationTrainingPipeline
)
from src.ML_emotion_detection.pipeline.Stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from src.ML_emotion_detection.pipeline.Stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from src.ML_emotion_detection.pipeline.Stage_05_model_evaluation import (
    ModelEvaluationPipeline,
)

# Data Ingestion Stage
STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\n", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

# Data Validation Stage
STAGE_NAME = "Data Validation stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\n", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

# Data Transformation Stage
STAGE_NAME = "Data Transformation stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\n", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

# Model Trainer Stage
STAGE_NAME = "Model Trainer stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    model_training = ModelTrainerTrainingPipeline()
    model_training.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\n", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e

# Model Evaluation Stage
STAGE_NAME = "Model evaluation stage"
try:
    logger.info(">>>>>> stage %s started <<<<<<", STAGE_NAME)
    model_evaluation = ModelEvaluationPipeline()
    model_evaluation.main()
    logger.info(">>>>>> stage %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.exception(e)
    raise e
