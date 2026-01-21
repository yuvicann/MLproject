import os
import sys
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlproject.logger import get_logger
from mlproject.exception import CustomException
from mlproject.utils.common import save_object

log = get_logger(__name__)


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline:
        - scale numeric columns
        - onehot encode categorical columns
        """
        try:
            cat_cols = X.select_dtypes(include="object").columns
            num_cols = X.select_dtypes(exclude="object").columns

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, raw_data_path: str):
        """
        Reads raw data, splits train/test,
        fits preprocessor on train, transforms both, saves preprocessor.

        Returns:
        X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_path
        """
        try:
            log.info("🔄 Data Transformation started")

            df = pd.read_csv(raw_data_path)

            target_col = "math_score"
            X = df.drop(columns=[target_col])
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            preprocessor = self.get_preprocessor(X_train)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(self.config.preprocessor_path, preprocessor)
            log.info(f"✅ Preprocessor saved at: {self.config.preprocessor_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(str(e), sys)
