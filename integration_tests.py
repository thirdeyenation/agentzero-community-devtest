import unittest
from agent_zero import AgentZero
from security import SecurityManager, PrivacyFilter
from explainable_ai import ExplainableAI

class AgentZeroIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.agent_zero = AgentZero()
        self.security_manager = SecurityManager()
        self.privacy_filter = PrivacyFilter()
        self.xai = ExplainableAI(self.agent_zero.model)

    def test_end_to_end_with_security(self):
        input_data = "What's the weather like in New York?"
        
        # Apply privacy filter
        filtered_input = self.privacy_filter.anonymize_data({"input": input_data})["input"]
        
        # Process through Agent Zero
        result = self.agent_zero.process_input(filtered_input)
        
        # Encrypt the result
        encrypted_result = self.security_manager.encrypt_data(result)
        
        # Decrypt and verify
        decrypted_result = self.security_manager.decrypt_data(encrypted_result)
        
        self.assertEqual(result, decrypted_result)
        self.assertIn("weather", decrypted_result.lower())
        self.assertIn("new york", decrypted_result.lower())

    def test_xai_integration(self):
        input_data = "Book a flight to London for tomorrow"
        
        # Process through Agent Zero
        result = self.agent_zero.process_input(input_data)
        
        # Generate explanation
        explanation = self.xai.generate_explanation(self.agent_zero.last_input_embedding)
        
        self.assertIsNotNone(explanation)
        self.assertIn("feature_importance", explanation)
        self.assertIn("top_features", explanation)

    def test_multi_agent_collaboration(self):
        complex_task = "Analyze the economic impact of renewable energy adoption in Europe"
        
        # Process through Agent Zero's multi-agent system
        result = self.agent_zero.multi_agent_system.collaborate(complex_task)
        
        self.assertIsNotNone(result)
        self.assertIn("economic impact", result.lower())
        self.assertIn("renewable energy", result.lower())
        self.assertIn("europe", result.lower())

if __name__ == "__main__":
    unittest.main()