import sys

from mlproject.logger import get_logger
from mlproject.exception import CustomException

from mlproject.components.data_ingestion import DataIngestion
from mlproject.components.data_transformation import DataTransformation
from mlproject.components.model_trainer import ModelTrainer

log = get_logger(__name__)


class TrainPipeline:
    def run_pipeline(self):
        try:
            log.info("🚀 Training Pipeline started")

            # ✅ Dataset path based on your structure
            input_csv_path = "notebooks/data/stud.csv"

            # 1) Data Ingestion
            ingestion = DataIngestion()
            raw_data_path = ingestion.initiate_data_ingestion(input_csv_path)

            # 2) Data Transformation
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test, preprocessor_path = transformation.initiate_data_transformation(
                raw_data_path
            )

            # 3) Model Training
            trainer = ModelTrainer()
            model_path, best_model_name, best_score = trainer.initiate_model_training(
                X_train, X_test, y_train, y_test
            )

            log.info("✅ Training Pipeline completed successfully")
            log.info(f"Best Model: {best_model_name}, R2 Score: {best_score}")
            log.info(f"Model saved at: {model_path}")
            log.info(f"Preprocessor saved at: {preprocessor_path}")

        except Exception as e:
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    TrainPipeline().run_pipeline()

