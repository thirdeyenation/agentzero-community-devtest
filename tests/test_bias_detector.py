import unittest
import numpy as np
from python.helpers.bias_detector import BiasDetector

class TestBiasDetector(unittest.TestCase):
    def setUp(self):
        self.bias_detector = BiasDetector()

    def test_no_bias(self):
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive_feature = np.array(['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'])
        self.assertTrue(self.bias_detector.check_bias(predictions, sensitive_feature))

    def test_bias_detected(self):
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive_feature = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        self.assertFalse(self.bias_detector.check_bias(predictions, sensitive_feature))

    def test_threshold(self):
        original_threshold = self.bias_detector.threshold
        self.bias_detector.threshold = 0.5
        predictions = np.array([1, 1, 1, 0, 0, 0])
        sensitive_feature = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        self.assertTrue(self.bias_detector.check_bias(predictions, sensitive_feature))
        self.bias_detector.threshold = original_threshold

if __name__ == '__main__':
    unittest.main()
