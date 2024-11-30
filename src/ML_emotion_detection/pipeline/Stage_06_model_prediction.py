"""
This module contains the PredictPipeline class for handling
emotion prediction using a pre-trained model and preprocessor.
"""

from src.ML_emotion_detection.utils.common import load_bin
from pathlib import Path
import pandas as pd
import numpy as np
import re


class PredictPipeline:
    """
    A class to manage emotion prediction using a pre-trained model
    and data preprocessing pipeline.
    """

    def __init__(self):
        """
        Initialize the PredictPipeline class with importing
        and downloading necessary downloads.

        Loads the pre-trained model and preprocessor from the specified
        file paths.
        """
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger_eng')
        self.model = load_bin(
            Path(
                "artifacts/model_trainer/log_reg_model.joblib"
            )
        )
        self.preprocessor = load_bin(
            Path("artifacts/data_transformation/preprocessor.joblib")
        )

    def validate_input(self, text):
        """
        Validate input text to ensure it is not numeric, empty,
        or lacks content.

        Args:
            text (str): Input text.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Check if the input is empty or None
        if not text or not text.strip():
            return False

        # Check if the input is fully numeric
        if text.strip().isdigit():
            return False

        # Check for at least one alphabetic character
        if not re.search(r'[a-zA-Z]', text):
            return False

        return True

    def predict(self, user_data):
        """
        Predict the emotion based on user input.

        Processes the input text using the preprocessor, performs prediction
        using the loaded model, and maps the predicted label to a
        human-readable emotion.

        Args:
            user_data (str): The input text provided by the user.

        Returns:
            str: A string containing the predicted emotion and its probability,
            or an error message for invalid input.

        Example:
            Input: "I am so happy today!"
            Output: "'joy' with probability of 0.98765"
        """
        if not isinstance(user_data, str):
            return "Invalid Input: Input must be a string."

        if not self.validate_input(user_data):
            return "Invalid Input: Text must contain meaningful content."

        # Convert the input text to a pandas Series
        data = pd.Series(user_data)

        # Transform the input text using the preprocessor
        processed_data = self.preprocessor.transform(data)

        # Predict the emotion label
        prediction = self.model.predict(processed_data)

        # Get the predicted probability
        pred_prob = "%.5f" % np.max(
            self.model.predict_proba(processed_data)
        )

        # Map the predicted label to a human-readable emotion
        label_map = {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
        prediction = label_map[prediction[0]]

        return f"{prediction} with probability of {pred_prob}'"
