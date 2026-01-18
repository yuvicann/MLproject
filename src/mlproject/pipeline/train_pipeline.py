import os
import sys
from dataclasses import dataclass

from mlproject.logger import get_logger
from mlproject.exception import CustomException
from mlproject.utils.common import save_object

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    artifacts_dir: str = "artifacts"
    model_path: str = os.path.join("artifacts", "model.pkl")


class TrainPipeline:
    def __init__(self):
        self.config = TrainingConfig()

    def run(self):
        """
        Full training pipeline steps (skeleton):
        1) Load / ingest data
        2) Preprocess / transform
        3) Train model
        4) Evaluate model
        5) Save model artifact
        """
        try:
            log.info("🚀 Training pipeline started")

            # ---------------------------
            # TODO: Replace this with real pipeline steps
            # ---------------------------
            dummy_model = {"model": "replace_with_real_model"}

            save_object(self.config.model_path, dummy_model)
            log.info(f"✅ Model saved at: {self.config.model_path}")

            log.info("🎉 Training pipeline completed successfully")

        except Exception as e:
            log.error("❌ Training pipeline failed", exc_info=True)
            raise CustomException(str(e), sys)


if __name__ == "__main__":
    TrainPipeline().run()
