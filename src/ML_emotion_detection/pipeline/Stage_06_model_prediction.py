from src.ML_emotion_detection.utils.common import load_bin
from pathlib import Path
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        self.model = load_bin(Path("artifacts/model_trainer/log_reg_model.joblib"))
        self.preprocessor = load_bin(
            Path("artifacts/data_transformation/preprocessor.joblib")
        )

    def predict(self, user_data):
        if isinstance(user_data, str):
            data = pd.Series(user_data)
            processed_data = self.preprocessor.transform(data)
            prediction = self.model.predict(processed_data)
            pred_prob = "%.5f" % np.max(self.model.predict_proba(processed_data))

            label_map = {
                0: "sadness",
                1: "joy",
                2: "love",
                3: "anger",
                4: "fear",
                5: "surprise",
            }

            prediction = label_map[prediction[0]]

            return f"'{prediction}' with probability of {pred_prob}"
        else:
            return "Invalid Input"
