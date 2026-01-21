from mlproject.pipeline.predict_pipeline import PredictPipeline, CustomData

if __name__ == "__main__":
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74,
    )

    df = data.get_data_as_dataframe()

    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    print("Predicted Math Score:", result[0])
