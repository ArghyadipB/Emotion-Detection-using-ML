from src.ML_emotion_detection.utils.common import load_bin
from src.ML_emotion_detection.pipeline.Stage_06_model_prediction \
    import PredictPipeline
import pandas as pd
import numpy as np
import time
import os

# load prediction pipeline
prediction_pipe = PredictPipeline()


# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))


# Load preprocessor
preprocessor_path = os.path.join(script_dir, 'artifacts',
                                 'data_transformation', 'preprocessor.joblib')
preprocessor = load_bin(preprocessor_path)


# load test
x_test_path = os.path.join(script_dir, 'artifacts',
                           'data_transformation', 'X_test.joblib')
x_test = load_bin(x_test_path)


def test_feature_extraction_pipeline():
    text = "I am happy because the day is bright"
    text = pd.Series(text)
    features = preprocessor.transform(text)
    assert features.shape == (1, 4017)


def test_prediction_pipeline():
    text = "happy sad angry @joy.l"
    prediction = prediction_pipe.predict(text).split()
    assert prediction[0] in ["sadness", "joy", "love",
                             "anger", "fear", "surprise"]


def test_invalid_input():
    input = 7896
    prediction = prediction_pipe.predict(input)
    assert prediction == "Invalid Input: Input must be a string."


def test_out_of_distribution():
    text = "!!!$$$???"
    prediction = prediction_pipe.predict(text)
    assert prediction == "Invalid Input: Text must contain meaningful content."


def test_long_text():
    text = "happy" * 100 + "emotional" * 200
    prediction = prediction_pipe.predict(text).split()
    assert prediction[0] in ["sadness", "joy", "love",
                             "anger", "fear", "surprise"]


def test_inference_time():
    input = pd.Series("I feel fantastic")
    start_time = time.time()
    prediction_pipe.predict(input)
    duration = time.time() - start_time
    assert duration < 1.0


def test_class_distribution():
    predictions = prediction_pipe.predict(x_test)
    unique, counts = np.unique(predictions, return_counts=True)
    assert all(count > 0 for count in counts)


def test_prediction_consistency():
    text = "I feel awesome"
    text = pd.Series(text)
    pred1 = prediction_pipe.predict(text)
    pred2 = prediction_pipe.predict(text)
    assert pred1 == pred2
