import unittest
from unittest.mock import patch
from io import StringIO
from python.helpers.human_verification import HumanVerification

class TestHumanVerification(unittest.TestCase):
    def setUp(self):
        self.human_verification = HumanVerification()

    def test_add_for_review(self):
        self.human_verification.add_for_review("Test content")
        self.assertEqual(self.human_verification.pending_reviews.qsize(), 1)

    @patch('builtins.input', side_effect=['y', 'n', 'y'])
    def test_review_content(self, mock_input):
        self.human_verification.add_for_review("Content 1")
        self.human_verification.add_for_review("Content 2")
        self.human_verification.add_for_review("Content 3")
        
        approved_content = self.human_verification.review_content()
        
        self.assertEqual(len(approved_content), 2)
        self.assertIn("Content 1", approved_content)
        self.assertIn("Content 3", approved_content)
        self.assertNotIn("Content 2", approved_content)

    def test_empty_review(self):
        approved_content = self.human_verification.review_content()
        self.assertEqual(len(approved_content), 0)

if __name__ == '__main__':
    unittest.main()