from fairlearn.metrics import demographic_parity_difference
import numpy as np

class BiasDetector:
    def __init__(self):
        self.threshold = 0.1

    def check_bias(self, predictions, sensitive_feature):
        y_true = np.ones_like(predictions)  # Assuming all predictions should be positive
        dpd = demographic_parity_difference(
            y_true=y_true,
            y_pred=predictions,
            sensitive_features=sensitive_feature
        )
        print(f"Demographic parity difference: {dpd}")
        return dpd < self.threshold