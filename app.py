from flask import Flask, request, jsonify, render_template

from mlproject.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    # UI page
    return render_template("index.html")


@app.route("/predict_form", methods=["POST"])
def predict_form():
    """
    Handles HTML form submission and shows prediction on UI.
    """
    try:
        gender = request.form.get("gender")
        race_ethnicity = request.form.get("race_ethnicity")
        parental_level_of_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_preparation_course = request.form.get("test_preparation_course")
        reading_score = int(request.form.get("reading_score"))
        writing_score = int(request.form.get("writing_score"))

        student_data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        input_df = student_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        result = round(float(prediction[0]), 2)

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


# ✅ Keep JSON API also
@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON prediction API
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

