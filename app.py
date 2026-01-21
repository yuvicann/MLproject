from flask import Flask, request, jsonify

from mlproject.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Performance Prediction API is running ✅"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON Input:
    {
      "gender": "female",
      "race_ethnicity": "group B",
      "parental_level_of_education": "bachelor's degree",
      "lunch": "standard",
      "test_preparation_course": "none",
      "reading_score": 72,
      "writing_score": 74
    }
    """
    try:
        data = request.get_json()

        student_data = CustomData(
            gender=data["gender"],
            race_ethnicity=data["race_ethnicity"],
            parental_level_of_education=data["parental_level_of_education"],
            lunch=data["lunch"],
            test_preparation_course=data["test_preparation_course"],
            reading_score=int(data["reading_score"]),
            writing_score=int(data["writing_score"]),
        )

        input_df = student_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        return jsonify({"predicted_math_score": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
