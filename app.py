from src.ML_emotion_detection.pipeline.Stage_06_model_prediction import \
    PredictPipeline
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
predict_pipe = PredictPipeline()


@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['user_input']
        prediction = predict_pipe.predict(user_input)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
