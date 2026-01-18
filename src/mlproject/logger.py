import logging
import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("mlproject")


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger for a specific module.
    Example: log = get_logger(__name__)
    """
    return logging.getLogger(name)
