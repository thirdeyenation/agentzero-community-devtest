from typing import List, Dict, Any

class EthicalFramework:
    def __init__(self):
        self.principles = {
            "beneficence": lambda action: action.get("positive_impact", 0) - action.get("negative_impact", 0),
            "non_maleficence": lambda action: -action.get("harm", 0),
            "autonomy": lambda action: action.get("user_choice", 0),
            "justice": lambda action: action.get("fairness", 0),
        }

    def evaluate_action(self, action: Dict[str, Any]) -> float:
        return sum(principle(action) for principle in self.principles.values())

class EthicalDecisionMaker:
    def __init__(self, ethical_framework: EthicalFramework):
        self.ethical_framework = ethical_framework

    def make_decision(self, possible_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        evaluated_actions = [(action, self.ethical_framework.evaluate_action(action)) for action in possible_actions]
        return max(evaluated_actions, key=lambda x: x[1])[0]

# Usage
ethical_framework = EthicalFramework()
decision_maker = EthicalDecisionMaker(ethical_framework)

possible_actions = [
    {"name": "Action A", "positive_impact": 5, "negative_impact": 2, "harm": 1, "user_choice": 3, "fairness": 4},
    {"name": "Action B", "positive_impact": 3, "negative_impact": 1, "harm": 0, "user_choice": 5, "fairness": 3},
    {"name": "Action C", "positive_impact": 7, "negative_impact": 4, "harm": 2, "user_choice": 2, "fairness": 5},
]

best_action = decision_maker.make_decision(possible_actions)
print("Most ethical action:", best_action["name"])