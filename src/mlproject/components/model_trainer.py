import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from mlproject.logger import get_logger
from mlproject.exception import CustomException
from mlproject.utils.common import save_object

log = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        """
        Trains multiple models and selects the best based on R2 score.
        Saves the best model into artifacts/model.pkl
        """
        try:
            log.info("🤖 Model Training started")

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, n_estimators=300),
                "CatBoost": CatBoostRegressor(verbose=False, random_state=42),
            }

            best_model_name = None
            best_model = None
            best_score = float("-inf")

            for name, model in models.items():
                log.info(f"Training model: {name}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                score = r2_score(y_test, y_pred)
                log.info(f"{name} R2 score: {score}")

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name

            save_object(self.config.model_path, best_model)
            log.info(f"✅ Best model saved: {best_model_name} at {self.config.model_path}")
            log.info(f"🏆 Best model R2 score: {best_score}")

            return self.config.model_path, best_model_name, best_score

        except Exception as e:
            raise CustomException(str(e), sys)
