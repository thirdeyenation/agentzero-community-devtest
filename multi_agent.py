import asyncio
from typing import List, Dict, Any

class Agent:
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Simulate processing time
        await asyncio.sleep(1)
        return {"agent": self.name, "result": f"Processed {task['type']} task using {self.specialty} specialty"}

class MultiAgentSystem:
    def __init__(self):
        self.agents: List[Agent] = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    async def collaborate(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async def process_task(task):
            suitable_agents = [agent for agent in self.agents if agent.specialty == task['required_specialty']]
            if not suitable_agents:
                return {"error": f"No suitable agent found for {task['type']} task"}
            return await suitable_agents[0].process_task(task)

        return await asyncio.gather(*(process_task(task) for task in tasks))

# Usage
multi_agent_system = MultiAgentSystem()
multi_agent_system.add_agent(Agent("Alice", "natural language processing"))
multi_agent_system.add_agent(Agent("Bob", "data analysis"))
multi_agent_system.add_agent(Agent("Charlie", "image recognition"))

async def main():
    tasks = [
        {"type": "text summarization", "required_specialty": "natural language processing"},
        {"type": "trend analysis", "required_specialty": "data analysis"},
        {"type": "object detection", "required_specialty": "image recognition"}
    ]
    results = await multi_agent_system.collaborate(tasks)
    print("Collaboration results:", results)

asyncio.run(main())