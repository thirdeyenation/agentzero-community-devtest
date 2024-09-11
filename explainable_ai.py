import shap
import numpy as np
from typing import List, Dict, Any

class ExplainableAI:
    def __init__(self, model):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def generate_explanation(self, input_data: np.ndarray) -> Dict[str, Any]:
        shap_values = self.explainer.shap_values(input_data)
        
        feature_importance = dict(zip(self.model.feature_names_, 
                                      np.abs(shap_values).mean(0)))
        
        top_features = sorted(feature_importance.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        
        explanation = {
            "feature_importance": feature_importance,
            "top_features": top_features,
            "shap_values": shap_values.tolist()
        }
        
        return explanation

    def generate_summary_plot(self, input_data: np.ndarray) -> None:
        shap.summary_plot(self.explainer.shap_values(input_data), 
                          input_data, feature_names=self.model.feature_names_)

    def generate_natural_language_explanation(self, explanation: Dict[str, Any]) -> str:
        nl_explanation = "The model's decision was primarily influenced by:\n"
        for feature, importance in explanation['top_features']:
            nl_explanation += f"- {feature}: contributing {importance:.2f} to the outcome\n"
        return nl_explanation

# Usage (assuming we have a trained model and input data)
# model = train_model()  # This would be your actual model training code
# input_data = prepare_input_data()  # This would be your actual input data

# xai = ExplainableAI(model)
# explanation = xai.generate_explanation(input_data)
# xai.generate_summary_plot(input_data)
# nl_explanation = xai.generate_natural_language_explanation(explanation)
# print(nl_explanation)