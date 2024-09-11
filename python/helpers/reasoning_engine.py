import networkx as nx
from typing import List, Dict, Any

class ReasoningEngine:
    def __init__(self):
        self.knowledge_graph = nx.DiGraph()

    def add_knowledge(self, subject: str, predicate: str, object: str):
        self.knowledge_graph.add_edge(subject, object, predicate=predicate)

    def reason(self, query: str) -> List[Dict[str, Any]]:
        start, end = query.split(' to ')
        try:
            path = nx.shortest_path(self.knowledge_graph, start, end)
            return [{"step": i, "from": path[i], "to": path[i+1], 
                     "relation": self.knowledge_graph[path[i]][path[i+1]]['predicate']}
                    for i in range(len(path)-1)]
        except nx.NetworkXNoPath:
            return []

class PlanningSystem:
    def __init__(self, reasoning_engine: ReasoningEngine):
        self.reasoning_engine = reasoning_engine

    def create_plan(self, goal: str, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        plan = []
        for key, target_value in goal.items():
            if current_state.get(key) != target_value:
                reasoning_result = self.reasoning_engine.reason(f"{current_state.get(key, 'unknown')} to {target_value}")
                plan.extend(reasoning_result)
        return plan
