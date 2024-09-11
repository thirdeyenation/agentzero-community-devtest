from agent_zero import AgentZero

if __name__ == "__main__":
    agent = AgentZero()
    agent.start()

    # Example usage
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = agent.process_input(user_input)
        print(f"Agent: {response}")
        
        # Collect feedback
        rating = int(input("Please rate the response (1-5): "))
        agent.collect_feedback("interaction_id", rating)  # You'd generate a real interaction_id in practice

    print("Goodbye!")