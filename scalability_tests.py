import concurrent.futures
import time
from agent_zero import AgentZero

class ScalabilityTester:
    def __init__(self, agent: AgentZero):
        self.agent = agent

    def run_concurrent_requests(self, num_requests: int, input_data: str):
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(self.agent.process_input, input_data) for _ in range(num_requests)]
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"Processed {num_requests} requests in {total_time:.2f} seconds")
        print(f"Average time per request: {total_time/num_requests:.2f} seconds")

    def test_increasing_load(self, max_requests: int, step: int):
        for num_requests in range(step, max_requests + 1, step):
            print(f"\nTesting with {num_requests} concurrent requests:")
            self.run_concurrent_requests(num_requests, "What's the weather like?")

# Usage
tester = ScalabilityTester(AgentZero())
tester.test_increasing_load(max_requests=100, step=20)