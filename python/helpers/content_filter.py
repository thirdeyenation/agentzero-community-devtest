from transformers import pipeline

class ContentFilter:
    def __init__(self):
        self.toxicity_model = pipeline("text-classification", model="unitary/toxic-bert")

    def filter_content(self, text: str) -> bool:
        results = self.toxicity_model(text)[0]
        return results['score'] > 0.5 and results['label'] == 'LABEL_1'  # toxic
