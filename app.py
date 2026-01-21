from flask import Flask, request, jsonify, render_template

from mlproject.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", result=None)


@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        student_data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=int(request.form.get("reading_score")),
            writing_score=int(request.form.get("writing_score")),
        )

        input_df = student_data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)

        result = round(float(prediction[0]), 2)
        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")


@app.route("/predict", methods=["POST"])
def predict_api():
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
    app.run(host="0.0.0.0", port=5001, debug=True)


