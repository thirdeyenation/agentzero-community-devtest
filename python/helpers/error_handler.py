import logging
from typing import Dict, Any

class ErrorHandler:
    def __init__(self, log_file: str = 'logs/error.log'):
        self.logger = logging.getLogger('error_handler')
        self.logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> str:
        error_message = f"Error: {str(error)}"
        if context:
            error_message += f" Context: {context}"
        self.logger.error(error_message, exc_info=True)
        return error_message

    def log_warning(self, message: str):
        self.logger.warning(message)
