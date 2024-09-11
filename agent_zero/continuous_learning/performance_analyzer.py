# File: agent_zero/continuous_learning/performance_analyzer.py

from typing import List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

class PerformanceAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.is_trained = False

    def preprocess_data(self, interactions: List[Dict[str, Any]], feedback: List[Dict[str, Any]]) -> tuple:
        """
        Preprocess the interaction and feedback data for model training.
        
        :param interactions: List of interaction data
        :param feedback: List of feedback data
        :return: Tuple of features (X) and labels (y)
        """
        # Combine interaction and feedback data
        combined_data = []
        for inter in interactions:
            for feed in feedback:
                if inter['id'] == feed['interaction_id']:
                    combined_data.append({
                        'input': inter['user_input'],
                        'response': inter['agent_response'],
                        'rating': feed['rating']
                    })
                    break
        
        # Extract features
        inputs = [item['input'] for item in combined_data]
        responses = [item['response'] for item in combined_data]
        
        X_input = self.vectorizer.fit_transform(inputs)
        X_response = self.vectorizer.transform(responses)
        X = np.hstack([X_input.toarray(), X_response.toarray()])
        
        y = np.array([item['rating'] for item in combined_data])
        
        return X, y

    def train_model(self, interactions: List[Dict[str, Any]], feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Train the performance prediction model.
        
        :param interactions: List of interaction data
        :param feedback: List of feedback data
        :return: Dictionary containing model performance metrics
        """
        X, y = self.preprocess_data(interactions, feedback)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_r2_score': train_score,
            'test_r2_score': test_score
        }

    def predict_satisfaction(self, user_input: str, agent_response: str) -> float:
        """
        Predict user satisfaction for a given interaction.
        
        :param user_input: User's input
        :param agent_response: Agent's response
        :return: Predicted satisfaction score
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train_model first.")
        
        X_input = self.vectorizer.transform([user_input])
        X_response = self.vectorizer.transform([agent_response])
        X = np.hstack([X_input.toarray(), X_response.toarray()])
        
        return self.model.predict(X)[0]

    def analyze_performance(self, interactions: List[Dict[str, Any]], feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the agent's performance.
        
        :param interactions: List of interaction data
        :param feedback: List of feedback data
        :return: Dictionary containing performance analysis results
        """
        total_interactions = len(interactions)
        average_rating = np.mean([f['rating'] for f in feedback])
        
        # Train the model and get performance metrics
        model_metrics = self.train_model(interactions, feedback)
        
        # Identify top and bottom performing interactions
        combined_data = []
        for inter in interactions:
            for feed in feedback:
                if inter['id'] == feed['interaction_id']:
                    combined_data.append({
                        'input': inter['user_input'],
                        'response': inter['agent_response'],
                        'rating': feed['rating']
                    })
                    break
        
        sorted_data = sorted(combined_data, key=lambda x: x['rating'], reverse=True)
        top_interactions = sorted_data[:5]
        bottom_interactions = sorted_data[-5:]
        
        return {
            'total_interactions': total_interactions,
            'average_rating': average_rating,
            'model_performance': model_metrics,
            'top_performing_interactions': top_interactions,
            'bottom_performing_interactions': bottom_interactions
        }

    def get_improvement_suggestions(self, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate improvement suggestions based on performance analysis.
        
        :param analysis_results: Results from analyze_performance method
        :return: List of improvement suggestions
        """
        suggestions = []
        
        if analysis_results['average_rating'] < 4.0:
            suggestions.append("Focus on improving overall response quality to increase average rating.")
        
        if analysis_results['model_performance']['test_r2_score'] < 0.6:
            suggestions.append("Collect more diverse feedback to improve prediction model accuracy.")
        
        for interaction in analysis_results['bottom_performing_interactions']:
            suggestions.append(f"Review and improve responses similar to: '{interaction['input'][:50]}...'")
        
        return suggestions