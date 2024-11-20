from src.ML_emotion_detection.config.configuration import ConfigurationManager
from src.ML_emotion_detection.components.data_transformation import DataTransformation
import pandas as pd
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


                Tr_X = Tr_X.progress_apply(data_transformation.preprocess)
                Te_X = Te_X.progress_apply(data_transformation.preprocess)


                Tr_pos_counts = Tr_X.progress_apply(data_transformation.pos_count_features)
                Te_pos_counts = Te_X.progress_apply(data_transformation.pos_count_features)


                Tr_X_intermediate = pd.concat([Tr_X, Tr_pos_counts], axis=1)
                Te_X_intermediate = pd.concat([Te_X, Te_pos_counts], axis=1)


                preprocessor = data_transformation.text_feature_extraction()
                data_transformation.transform_and_save(preprocessor, Tr_X_intermediate, Te_X_intermediate)
                
        except Exception as e:
            raise e