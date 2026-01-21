import sys
import pandas as pd

from mlproject.logger import get_logger
from mlproject.exception import CustomException
from mlproject.utils.common import load_object

log = get_logger(__name__)


class PredictPipeline:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame):
        """
        Takes input features dataframe, applies preprocessing, and returns prediction.
        """
        try:
            log.info("🔮 Prediction started")

            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            transformed_features = preprocessor.transform(features)
            preds = model.predict(transformed_features)

            log.info("✅ Prediction completed")
            return preds

        except Exception as e:
            raise CustomException(str(e), sys)


class CustomData:
    """
    This class converts user input into a DataFrame for prediction.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(str(e), sys)
