import os
import sys
import pandas as pd
from dataclasses import dataclass

from mlproject.logger import get_logger
from mlproject.exception import CustomException

log = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self, input_csv_path: str) -> str:
        """
        Reads the input CSV and saves a raw copy into artifacts folder.
        Returns the raw data path.
        """
        try:
            log.info("📥 Data Ingestion started")

            df = pd.read_csv(input_csv_path)

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            log.info(f"✅ Raw data saved at: {self.config.raw_data_path}")
            return self.config.raw_data_path

        except Exception as e:
            raise CustomException(str(e), sys)
