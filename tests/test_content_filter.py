import unittest
from python.helpers.content_filter import ContentFilter

class TestContentFilter(unittest.TestCase):
    def setUp(self):
        self.content_filter = ContentFilter()

    def test_filter_toxic_content(self):
        toxic_text = "You are a terrible person!"
        self.assertTrue(self.content_filter.filter_content(toxic_text))

    def test_allow_non_toxic_content(self):
        non_toxic_text = "Have a nice day!"
        self.assertFalse(self.content_filter.filter_content(non_toxic_text))

    def test_edge_case_neutral_content(self):
        neutral_text = "The sky is blue."
        self.assertFalse(self.content_filter.filter_content(neutral_text))

if __name__ == '__main__':
    unittest.main()
