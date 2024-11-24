from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.data_transformation import DataTransformation
from pathlib import Path


STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
           
                data_transformation = DataTransformation(config=data_transformation_config)
                Tr_X, Te_X = data_transformation.train_test_spliting()
                data_transformation.pipeline_and_transform(Tr_X, Te_X)
                                
        except Exception as e:
            raise e