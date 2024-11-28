"""
This module contains the Flask application for serving the emotion
detection model. It handles user inputs through a web interface
and provides predictions via a REST API.
"""

from src.ML_emotion_detection.pipeline.Stage_06_model_prediction import \
    PredictPipeline
from flask import Flask, render_template, request, jsonify

# Initialize the Flask application and prediction pipeline
app = Flask(__name__)
predict_pipe = PredictPipeline()


@app.route("/", methods=["GET"])
def homePage():
    """
    Render the homepage.

    Returns:
        HTML: Renders the `index.html` template for the home page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the prediction request.

    Accepts user input through an HTML form, processes it using the
    prediction pipeline, and returns the predicted emotion as JSON.

    Returns:
        JSON: Contains the prediction result or an error message
        in case of failure.

    Example Response:
        {
            "prediction": "happy"
        }
        or
        {
            "error": "Error message"
        }
    """
    try:
        # Get user input from the form
        user_input = request.form["user_input"]

        # Generate prediction
        prediction = predict_pipe.predict(user_input)

        # Return the prediction as JSON
        return jsonify({"prediction": prediction})
    except Exception as e:
        # Handle any errors and return the error message as JSON
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    """
    Run the Flask application in debug mode.
    """
    app.run(debug=True)
