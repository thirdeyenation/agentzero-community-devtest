# Agent Zero

[![Join our Skool Community](https://img.shields.io/badge/Skool-Join%20our%20Community-4A90E2?style=for-the-badge&logo=skool&logoColor=white)](https://www.skool.com/agent-zero) [![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/B8KZKNsPpj) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AgentZeroFW) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jan-tomasek/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/JanTomasekDev)


[![Intro Video](/docs/res/intro_vid.jpg)](https://www.youtube.com/watch?v=C9n8zFpaV3I)

**Personal and organic AI framework**
- Agent Zero is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
- Agent Zero is fully transparent, readable, comprehensible, customizable and interactive.
- Agent Zero uses the computer as a tool to accomplish its (your) tasks.

## Now with UI:
![UI prototype](/docs/res/ui_screen1.png)

## Key concepts
1. **General-purpose assistant**
- Agent Zero is not pre-programmed for specific tasks (but can be). It is meant to be a general-purpose personal assistant. Give it a task, and it will gather information, execute commands and code, cooperate with other agent instances, and do its best to accomplish it.
- It has a persistent memory, allowing it to memorize previous solutions, code, facts, instructions, etc., to solve tasks faster and more reliably in the future.

2. **Computer as a tool**
- Agent Zero uses the operating system as a tool to accomplish its tasks. It has no single-purpose tools pre-programmed. Instead, it can write its own code and use the terminal to create and use its own tools as needed.
- The only default tools in its arsenal are online search, memory features, communication (with the user and other agents), and code/terminal execution. Everything else is created by the agent itself or can be extended by the user.
- Tool usage functionality has been developed from scratch to be the most compatible and reliable, even with very small models.

3. **Multi-agent cooperation**
- Every agent has a superior agent giving it tasks and instructions. Every agent then reports back to its superior.
- In the case of the first agent in the chain (Agent 0), the superior is the human user; the agent sees no difference.
- Every agent can create its subordinate agent to help break down and solve subtasks. This helps all agents keep their context clean and focused.

4. **Completely customizable and extensible**
- Almost nothing in this framework is hard-coded. Nothing is hidden. Everything can be extended or changed by the user.
- The whole behavior is defined by a system prompt in the **prompts/default/agent.system.md** file. Change this prompt and change the framework dramatically.
- The framework does not guide or limit the agent in any way. There are no hard-coded rails that agents have to follow.
- Every prompt, every small message template sent to the agent in its communication loop, can be found in the **prompts/** folder and changed.
- Every default tool can be found in the **python/tools/** folder and changed or copied to create new predefined tools.
- Of course, it is open-source (except for some tools like Perplexity, but that will be replaced with an open-source alternative as well in the future).

5. **Communication is key**
- Give your agent a proper system prompt and instructions, and it can do miracles.
- Agents can communicate with their superiors and subordinates, asking questions, giving instructions, and providing guidance. Instruct your agents in the system prompt on how to communicate effectively.
- The terminal interface is real-time streamed and interactive. You can stop and intervene at any point. If you see your agent heading in the wrong direction, just stop and tell it right away.
- There is a lot of freedom in this framework. You can instruct your agents to regularly report back to superiors asking for permission to continue. You can instruct them to use point-scoring systems when deciding when to delegate subtasks. Superiors can double-check subordinates' results and dispute. The possibilities are endless.

![Agent Zero](/docs/res/splash_wide.png)

## Nice features to have
- Output is very clean, colorful, readable and interactive; nothing is hidden.
- The same colorful output you see in the terminal is automatically saved to HTML file in **logs/** folder for every session.
- Agent output is streamed in real-time, allowing the user to read along and intervene at any time.
- No coding is required, only prompting and communication skills.
- With a solid system prompt, the framework is reliable even with small models, including precise tool usage.

## Keep in mind
1. **Agent Zero can be dangerous!**
With proper instruction, Agent Zero is capable of many things, even potentially dangerous to your computer, data, or accounts. Always run Agent Zero in an isolated environment (like the built in docker container) and be careful what you wish for.

2. **Agent Zero is not pre-programmed; it is prompt-based.**
The whole framework contains only a minimal amount of code and does not guide the agent in any way.
Everything lies in the system prompt in the **prompts/** folder. Here you can rewrite the whole framework behavior to your needs.
If your agent fails to communicate properly, use tools, reason, use memory, find answers - just instruct it better.

3. **If you cannot provide the ideal environment, let your agent know.**
Agent Zero is made to be used in an isolated virtual environment (for safety) with some tools preinstalled and configured.
If you cannot provide all the necessary conditions or API keys, just change the system prompt and tell your agent what operating system and tools are at its disposal. Nothing is hard-coded; if you do not tell your agent about a certain tool, it will not know about it and will not try to use it.


[![David Ondrej video](/docs/res/david_vid.jpg)](https://www.youtube.com/watch?v=_Pionjv4hGc)

## Known problems
1. The system prompt sucks. You can do better. If you do, help me please :)
2. The communication between agent and terminal in docker container via SSH can sometimes break and stop producing outputs. Sometimes it is because the agent runs something like "server.serve_forever()" which causes the terminal to hang, sometimes a random error can occur. Restarting the agent and/or the docker container helps.
3. The agent can break his operating system. Sometimes the agent can deactivate virtual environment, uninstall packages, change config etc. Again, removing the docker container and cleaning up the **work_dir/** is enough to fix that.

## Ideal environment
- **Docker container**: The perfect environment to run Agent Zero is the built-in docker container. The agent can download the image **frdel/agent-zero-exe** on its own and start the container, you only need to have docker running (like the Docker Desktop application).
- **Python**: Python has to be installed on the system to run the framework.
- **Internet access**: The agent will need internet access to use its online knowledge tool and execute commands and scripts requiring a connection. If you do not need your agent to be online, you can alter its prompts in the **prompts/** folder and make it fully local.

![Time example](/docs/res/time_example.jpg)

## Setup

A detailed setup guide with a video can be found here: [/docs/installation](https://github.com/frdel/agent-zero/tree/main/docs/installation). Scroll down to see the readme file.

> ⚠️ **Changes to launch files since v0.6:**  
> main.py file has been replaced with run_ui.py (webui) and run_cli.py (terminal) launch files.
> configuration has been moved to initialize.py for both webui and terminal launch files.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ############################################ #
###         **Agent Zero Dev. Branch**       ###  
### CHANGES, NOTES, & IMPORTANT INSTRUCTIONS ###
# ############################################ #

# NOTES & IMPORTANT INSTRUCTIONS

1) **Enhanced Integration Testing**
I've implemented a comprehensive suite of integration tests to ensure the proper interaction between various components of Agent Zero.
# Features:
- Tests for interaction between Agent, ContentFilter, HumanVerification, BiasDetector, and ErrorHandler
- Scenarios covered include content filtering, bias detection, and error handling
- Mock objects used for external dependencies to focus on inter-component interactions
#### Running Integration Tests:
To run the integration tests, use the following command from the project root:

```
python -m unittest tests.test_integration
```

These tests are crucial for maintaining the reliability and correctness of Agent Zero as the system evolves. They should be run after any significant changes to ensure that all components still work together as expected.

# RECENT CHANGES 
1) 
- Added agent_zero folder to Root Directory. -> /Agent-Zero-0.6.2/agent_zero 
- Within agent_zero folder: 
  vector_db.py, 
  security.py, 
  knowledge_base.py, 
  config,py, 
  _init_.py. 
- Within agent_zero folder, find sub-folder, continuous_learning. -> /Agent-Zero-0.6.2/agent_zero/continuous_learning 
- Within continuous_learning sub-folder: 
  performance_anaylzer.py, 
  knowldge_integrator.py, 
  interacction_logger.py, 
  human_verification.py, 
  feedback_collector.py, 
  data_collecter.py, 
  bias_detector.py, 
  _init_.py 

2) 
- Updated agent.tools.md prompt

3) 
- Updated agent.system.md prompt

4) 
- Added files to /Agent-Zero-0.6.2/python/helpers
  bias.detector.py
  content_filter.py
  human_verification.py
  reasoning_engine.py

5) 
- Added "tests" folder to Root Directory. -> /Agent-Zero-0.6.2/tests
- Sub-folder "helpers" added to "tests" folder. -> /Agent-Zero-0.6.2/tests/helpers
  Files added to "helpers" sub-folder: 
  _init_.py, 
  human_verification.py, 
  test_bias_detector.py, 
  test_content_filter.py, 
  test_error_handler.py, 
  test_integration.py, 
  test_reasoning_engine.py

6) 
- Added files to Root Directory: 
  cognitive_architecture.py
  continual_learning.py
  dynamic_tools.py
  ethics.py
  explainable_ai.py
  integration_tests.py
  orchestrator.py
  multi_agent.py
  models.py
  performance_optimizer.py
  reasoning.py
  scalability_tests.py
  security.py

# CHANGES
1) Added Roadmap in -> /docs/build/doctrees/RoadMap 

## Next Steps
- Develop advanced reasoning capabilities
- Enhance multi-modal interaction support
- Implement continuous learning mechanisms

...

# ISSUES
