from src.ML_emotion_detection.entity.config_entity import DataValidationConfig
from src.ML_emotion_detection import logger
import pandas as pd

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_parquet(self.config.unzip_data_dir,engine="pyarrow")
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            logger.info("Checking if the required columns exist or not")
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e