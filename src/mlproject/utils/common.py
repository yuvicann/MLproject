import os
import pickle
from typing import Any


def create_directories(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def save_object(file_path: str, obj: Any) -> None:
    """
    Save any Python object using pickle.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        create_directories(dir_path)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: str) -> Any:
    """
    Load a pickled object.
    """
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)
