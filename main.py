from flask import Flask, jsonify, request
from typing import Literal
from models.data_processor import DataPreprocessor
from models import ModelTrainer


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]



# Load model and vectorizer
vectorizer_path = 'models/vectorizer.pkl'
model_path = 'models/model.pkl'

data_processor = DataPreprocessor(file_path=None)  # file_path not needed for inference
data_processor.load_vectorizer(vectorizer_path)

trainer = ModelTrainer()
trainer.load_model(model_path)


def predict(description: str) -> LABELS:
    """
    Function that should take in the description text and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """
    try:

        features = data_processor.extract_features(description)

        # Predict
        prediction = trainer.predict(features)
        return str(prediction[0])



    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)

    prediction = predict(data["description"])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)