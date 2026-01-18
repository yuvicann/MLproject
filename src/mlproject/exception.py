import sys
import traceback
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts error details like file name, line number, and message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown file"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    return f"Error in file [{file_name}] at line [{line_number}] : {str(error)}"


class CustomException(Exception):
    """
    Custom Exception class to wrap errors with detailed trace.
    """

    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            Exception(error_message), error_detail if error_detail else sys
        )

    def __str__(self):
        return self.error_message
