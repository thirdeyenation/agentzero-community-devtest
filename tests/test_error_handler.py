import unittest
from unittest.mock import patch, MagicMock
from python.helpers.error_handler import ErrorHandler

class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.error_handler = ErrorHandler()

    @patch('logging.Logger.error')
    def test_handle_error(self, mock_error):
        error = ValueError("Test error")
        context = {"key": "value"}
        result = self.error_handler.handle_error(error, context)
        mock_error.assert_called_once()
        self.assertIn("Test error", result)
        self.assertIn("Context: {'key': 'value'}", result)

    @patch('logging.Logger.warning')
    def test_log_warning(self, mock_warning):
        self.error_handler.log_warning("Test warning", {"key": "value"})
        mock_warning.assert_called_once_with("Warning: Test warning Context: {'key': 'value'}")

    @patch('logging.Logger.info')
    def test_log_info(self, mock_info):
        self.error_handler.log_info("Test info")
        mock_info.assert_called_once_with("Info: Test info")

    @patch('logging.Logger.error')
    def test_handle_error_without_context(self, mock_error):
        error = ValueError("Test error")
        result = self.error_handler.handle_error(error)
        mock_error.assert_called_once()
        self.assertIn("Test error", result)
        self.assertNotIn("Context:", result)

if __name__ == '__main__':
    unittest.main()
