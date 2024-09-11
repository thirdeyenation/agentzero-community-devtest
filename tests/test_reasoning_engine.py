import unittest
from python.helpers.reasoning_engine import ReasoningEngine, PlanningSystem

class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.reasoning_engine = ReasoningEngine()
        self.reasoning_engine.add_knowledge("A", "connects to", "B")
        self.reasoning_engine.add_knowledge("B", "leads to", "C")

    def test_reason(self):
        result = self.reasoning_engine.reason("A to C")
        expected = [
            {"step": 0, "from": "A", "to": "B", "relation": "connects to"},
            {"step": 1, "from": "B", "to": "C", "relation": "leads to"}
        ]
        self.assertEqual(result, expected)

class TestPlanningSystem(unittest.TestCase):
    def setUp(self):
        self.reasoning_engine = ReasoningEngine()
        self.planning_system = PlanningSystem(self.reasoning_engine)

    def test_create_plan(self):
        self.reasoning_engine.add_knowledge("New York", "is in", "USA")
        self.reasoning_engine.add_knowledge("USA", "has capital", "Washington D.C.")
        
        current_state = {"location": "New York"}
        goal = {"location": "Washington D.C."}
        
        plan = self.planning_system.create_plan(goal, current_state)
        expected = [
            {"step": 0, "from": "New York", "to": "USA", "relation": "is in"},
            {"step": 1, "from": "USA", "to": "Washington D.C.", "relation": "has capital"}
        ]
        self.assertEqual(plan, expected)

class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.reasoning_engine = ReasoningEngine()
        self.reasoning_engine.add_knowledge("A", "connects to", "B")
        self.reasoning_engine.add_knowledge("B", "leads to", "C")
        self.reasoning_engine.add_knowledge("C", "contains", "D")

    def test_add_knowledge(self):
        self.reasoning_engine.add_knowledge("E", "relates to", "F")
        self.assertIn("E", self.reasoning_engine.knowledge_graph.nodes())
        self.assertIn("F", self.reasoning_engine.knowledge_graph.nodes())
        self.assertIn("relates to", self.reasoning_engine.knowledge_graph["E"]["F"])

    def test_reason_valid_path(self):
        result = self.reasoning_engine.reason("A to D")
        expected = [
            {"step": 0, "from": "A", "to": "B", "relation": "connects to"},
            {"step": 1, "from": "B", "to": "C", "relation": "leads to"},
            {"step": 2, "from": "C", "to": "D", "relation": "contains"}
        ]
        self.assertEqual(result, expected)

    def test_reason_invalid_path(self):
        result = self.reasoning_engine.reason("A to F")
        self.assertEqual(result, [])

class TestPlanningSystem(unittest.TestCase):
    def setUp(self):
        self.reasoning_engine = ReasoningEngine()
        self.reasoning_engine.add_knowledge("New York", "is in", "USA")
        self.reasoning_engine.add_knowledge("USA", "has capital", "Washington D.C.")
        self.reasoning_engine.add_knowledge("Washington D.C.", "has monument", "Lincoln Memorial")
        self.planning_system = PlanningSystem(self.reasoning_engine)

    def test_create_plan(self):
        current_state = {"location": "New York"}
        goal = {"location": "Lincoln Memorial"}
        plan = self.planning_system.create_plan(goal, current_state)
        expected = [
            {"step": 0, "from": "New York", "to": "USA", "relation": "is in"},
            {"step": 1, "from": "USA", "to": "Washington D.C.", "relation": "has capital"},
            {"step": 2, "from": "Washington D.C.", "to": "Lincoln Memorial", "relation": "has monument"}
        ]
        self.assertEqual(plan, expected)

    def test_create_plan_no_change_needed(self):
        current_state = {"location": "Lincoln Memorial"}
        goal = {"location": "Lincoln Memorial"}
        plan = self.planning_system.create_plan(goal, current_state)
        self.assertEqual(plan, [])

    def test_create_plan_unreachable_goal(self):
        current_state = {"location": "New York"}
        goal = {"location": "Moon"}
        plan = self.planning_system.create_plan(goal, current_state)
        self.assertEqual(plan, [])

if __name__ == '__main__':
    unittest.main()