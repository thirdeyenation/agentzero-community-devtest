"""
agent.py - Core implementation of Agent Zero

This module contains the main Agent class and supporting classes for
managing context, configuration, and continuous learning capabilities.
"""

import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
import uuid

# Third-party imports
from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings

# Local imports
import python.helpers.log as Log
from python.helpers.dirty_json import DirtyJson
from python.helpers.defer import DeferredTask
from python.helpers import extract_tools, rate_limiter, files, errors
from python.helpers.print_style import PrintStyle
from python.helpers.content_filter import ContentFilter
from python.helpers.human_verification import HumanVerification
from python.helpers.bias_detector import BiasDetector
from python.helpers.error_handler import ErrorHandler
from python.helpers.reasoning_engine import ReasoningEngine, PlanningSystem

# Continuous Learning Pipeline components
from agent_zero.continuous_learning.data_collector import DataCollector
from agent_zero.continuous_learning.knowledge_integrator import KnowledgeIntegrator
from agent_zero.continuous_learning.interaction_logger import InteractionLogger
from agent_zero.continuous_learning.feedback_collector import FeedbackCollector
from agent_zero.continuous_learning.performance_analyzer import PerformanceAnalyzer

from agent_zero.knowledge_base import KnowledgeBase
from agent_zero.vector_db import VectorDB


class AgentContext:
    """Manages the context for Agent Zero instances."""

    _contexts: Dict[str, "AgentContext"] = {}
    _counter: int = 0

    def __init__(
        self,
        config: "AgentConfig",
        id: Optional[str] = None,
        agent0: Optional["Agent"] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.config = config
        self.log = Log.Log()
        self.agent0 = agent0 or Agent(0, self.config, self)
        self.paused = False
        self.streaming_agent: Optional[Agent] = None
        self.process: Optional[DeferredTask] = None
        AgentContext._counter += 1
        self.no = AgentContext._counter

        self._contexts[self.id] = self

    @staticmethod
    def get(id: str) -> Optional["AgentContext"]:
        """Retrieve an AgentContext by its ID."""
        return AgentContext._contexts.get(id)

    @staticmethod
    def first() -> Optional["AgentContext"]:
        """Retrieve the first AgentContext if any exist."""
        return next(iter(AgentContext._contexts.values()), None)

    @staticmethod
    def remove(id: str) -> Optional["AgentContext"]:
        """Remove and return an AgentContext by its ID."""
        context = AgentContext._contexts.pop(id, None)
        if context and context.process:
            context.process.kill()
        return context

    def reset(self):
        """Reset the AgentContext to its initial state."""
        if self.process:
            self.process.kill()
        self.log.reset()
        self.agent0 = Agent(0, self.config, self)
        self.streaming_agent = None
        self.paused = False

    def communicate(self, msg: str, broadcast_level: int = 1) -> DeferredTask:
        """Communicate a message to the agent."""
        self.paused = False

        if self.process and self.process.is_alive():
            current_agent = self.streaming_agent or self.agent0
            intervention_agent = current_agent
            while intervention_agent and broadcast_level != 0:
                intervention_agent.intervention_message = msg
                broadcast_level -= 1
                intervention_agent = intervention_agent.data.get("superior", None)
        else:
            self.process = DeferredTask(self.agent0.message_loop, msg)

        return self.process


@dataclass
class AgentConfig:
    """Stores the configuration settings for an Agent Zero instance."""

    chat_model: BaseChatModel | BaseLLM
    utility_model: BaseChatModel | BaseLLM
    embeddings_model: Embeddings
    prompts_subdir: str = ""
    memory_subdir: str = ""
    knowledge_subdir: str = ""
    auto_memory_count: int = 3
    auto_memory_skip: int = 2
    rate_limit_seconds: int = 60
    rate_limit_requests: int = 15
    rate_limit_input_tokens: int = 0
    rate_limit_output_tokens: int = 0
    msgs_keep_max: int = 25
    msgs_keep_start: int = 5
    msgs_keep_end: int = 10
    response_timeout_seconds: int = 60
    max_tool_response_length: int = 3000
    code_exec_docker_enabled: bool = True
    code_exec_docker_name: str = "agent-zero-exe"
    code_exec_docker_image: str = "frdel/agent-zero-exe:latest"
    code_exec_docker_ports: dict[str, int] = field(default_factory=lambda: {"22/tcp": 50022})
    code_exec_docker_volumes: dict[str, dict[str, str]] = field(default_factory=lambda: {files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"}})
    code_exec_ssh_enabled: bool = True
    code_exec_ssh_addr: str = "localhost"
    code_exec_ssh_port: int = 50022
    code_exec_ssh_user: str = "root"
    code_exec_ssh_pass: str = "toor"
    additional: Dict[str, Any] = field(default_factory=dict)


class InterventionException(Exception):
    """An exception raised to signal an intervention during the agent's message loop."""
    pass


class KillerException(Exception):
    """A critical exception that terminates the agent's message loop."""
    pass


class Agent:
    """The core class representing an Agent Zero instance."""

    def __init__(
        self, number: int, config: AgentConfig, context: Optional[AgentContext] = None
    ):
        self.number = number
        self.config = config
        self.context = context or AgentContext(config)
        self.agent_name = f"Agent {self.number}"
        self.history = []
        self.last_message = ""
        self.intervention_message = ""
        self.rate_limiter = rate_limiter.RateLimiter(
            self.context.log,
            max_calls=self.config.rate_limit_requests,
            max_input_tokens=self.config.rate_limit_input_tokens,
            max_output_tokens=self.config.rate_limit_output_tokens,
            window_seconds=self.config.rate_limit_seconds,
        )
        self.data = {}
        self.memory_skip_counter = 0

        # Continuous Learning Pipeline components
        self.knowledge_base = KnowledgeBase()
        self.vector_db = VectorDB()
        self.data_collector = DataCollector()
        self.knowledge_integrator = KnowledgeIntegrator(self.knowledge_base, self.vector_db)
        self.interaction_logger = InteractionLogger()
        self.feedback_collector = FeedbackCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.content_filter = ContentFilter()
        self.human_verification = HumanVerification()
        self.bias_detector = BiasDetector()
        self.error_handler = ErrorHandler()
        self.recent_predictions = []
        self.sensitive_features = []
        self.reasoning_engine = ReasoningEngine()
        self.planning_system = PlanningSystem(self.reasoning_engine)

    async def process_input(self, user_input: str) -> tuple[str, str]:
        """
        Process user input, trigger the message loop, log the interaction,
        and return the agent's response and the interaction ID.
        """
        response = await self.message_loop(user_input)
        interaction_id = self.interaction_logger.log_interaction(user_input, response)
        return response, interaction_id

    async def generate_response(self, input_data: str, sensitive_feature: str) -> str:
        """Generate a response based on the input data and sensitive feature."""
        try:
            if self.content_filter.filter_content(input_data):
                return "I'm sorry, but I can't process that input due to inappropriate content."

            current_state = self.extract_current_state(input_data)
            goal = self.extract_goal(input_data)
            plan = self.planning_system.create_plan(goal, current_state)
            
            response = await self.execute_plan(plan)
            
            self.recent_predictions.append(1 if response else 0)
            self.sensitive_features.append(sensitive_feature)
            
            if len(self.recent_predictions) >= 100:
                if not self.bias_detector.check_bias(self.recent_predictions, self.sensitive_features):
                    self.error_handler.log_warning("Bias detected. Initiating mitigation strategies.")
                self.recent_predictions = []
                self.sensitive_features = []
            
            return response
        except Exception as e:
            error_message = self.error_handler.handle_error(e, {"input_data": input_data, "sensitive_feature": sensitive_feature})
            return f"An error occurred while generating the response: {error_message}"

    async def message_loop(self, msg: str):
        """The main message processing loop for the agent."""
        try:
            printer = PrintStyle(italic=True, font_color="#b3ffd9", padding=False)    
            user_message = self.read_prompt("fw.user_message.md", message=msg)
            await self.append_message(user_message, human=True)
            memories = await self.fetch_memories(True)
            
            while True:
                self.context.streaming_agent = self
                agent_response = ""

                try:
                    current_state = self.extract_current_state(msg)
                    goal = self.extract_goal(msg)
                    plan = self.planning_system.create_plan(goal, current_state)
                    
                    reasoning_prompt = f"Based on the current state: {current_state} and the goal: {goal}, consider this plan: {plan}"
                    prompt = self.read_prompt("agent.system.md", agent_name=self.agent_name)
                    prompt += "\n\n" + self.read_prompt("agent.tools.md")
                    prompt += f"\n\n{reasoning_prompt}"

                    if memories:
                        prompt += "\n\n" + memories

                    chat_prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(content=prompt),
                        MessagesPlaceholder(variable_name="messages"),
                    ])

                    inputs = {"messages": self.history}
                    chain = chat_prompt | self.config.chat_model

                    formatted_inputs = chat_prompt.format(messages=self.history)
                    tokens = int(len(formatted_inputs) / 4)
                    self.rate_limiter.limit_call_and_input(tokens)
                    
                    PrintStyle(
                        bold=True,
                        font_color="green",
                        padding=True,
                        background_color="white",
                    ).print(f"{self.agent_name}: Generating:")
                    log = self.context.log.log(
                        type="agent", heading=f"{self.agent_name}: Generating:"
                    )
                              
                    async for chunk in chain.astream(inputs):
                        await self.handle_intervention(agent_response)

                        content = str(chunk.content) if hasattr(chunk, "content") else str(chunk)
                        
                        if content:
                            printer.stream(content)
                            agent_response += content
                            self.log_from_stream(agent_response, log)

                    self.rate_limiter.set_output_tokens(int(len(agent_response) / 4))
                    
                    await self.handle_intervention(agent_response)

                    if self.last_message == agent_response:
                        await self.append_message(agent_response)
                        warning_msg = self.read_prompt("fw.msg_repeat.md")
                        await self.append_message(warning_msg, human=True)
                        PrintStyle(font_color="orange", padding=True).print(warning_msg)
                        self.context.log.log(type="warning", content=warning_msg)
                    else:
                        await self.append_message(agent_response)
                        tools_result = await self.process_tools(agent_response)
                        if tools_result:
                            return tools_result

                except InterventionException:
                    pass
                except asyncio.CancelledError as e:
                    PrintStyle(
                        font_color="white", background_color="red", padding=True
                    ).print(f"Context {self.context.id} terminated during message loop")
                    raise e
                except KillerException as e:
                    error_message = errors.format_error(e)
                    self.context.log.log(type="error", content=error_message)
                    raise e
                except Exception as e:
                    error_message = errors.format_error(e)
                    msg_response = self.read_prompt("fw.error.md", error=error_message)
                    await self.append_message(msg_response, human=True)
                    PrintStyle(font_color="red", padding=True).print(msg_response)
                    self.context.log.log(type="error", content=msg_response)
        
        finally:
            self.context.streaming_agent = None

    async def continuous_learning_cycle(self):
        """The continuous learning cycle for the agent."""
        while True:
            new_data = await self.data_collector.collect()

            filtered_data = [
                d for d in new_data
                if not self.content_filter.filter_content(d["content"])
            ]
            for item in filtered_data:
                self.human_verification.add_for_review(item["content"])
            verified_data = self.human_verification.review_content()
            
            self.knowledge_integrator.integrate_new_knowledge(verified_data)

            interactions = self.interaction_logger.get_logs()
            feedback = self.feedback_collector.get_feedback()
            self.performance_analyzer.train_model(interactions, feedback)

            recent_predictions = [
                self.performance_analyzer.predict_satisfaction(i)
                for i in interactions[-100:]
            ]
            sensitive_features = [i.split("|")[0] for i in interactions[-100:]]
            if not self.bias_detector.check_bias(recent_predictions, sensitive_features):
                print("Bias detected. Initiating mitigation strategies.")
                # TODO: Implement bias mitigation strategies here

            self.knowledge_integrator.create_snapshot()

            await asyncio.sleep(86400)  # Run daily

    def extract_current_state(self, input_data: str) -> Dict[str, Any]:
        """Extract the current state from the input data."""
        return {"input": input_data}

    def extract_goal(self, input_data: str) -> Dict[str, Any]:
        """Extract the goal from the input data."""
        return {"response": "generated"}

    async def execute_plan(self, plan: List[Dict[str, Any]]) -> str:
        """Execute the plan and generate a response."""
        return f"Executed plan based on input: {plan}"

    def collect_user_feedback(
        self, 
        interaction_id: str, 
        user_rating: int, 
        user_comment: Optional[str] = None
    ) -> None:
        """Collect feedback on a specific interaction."""
        self.feedback_collector.collect_feedback(interaction_id, user_rating, user_comment)

    def collect_feedback_from_system(
        self, interaction_id: str, system_rating: int, system_comment: Optional[str] = None
    ) -> None:
        """Collect feedback from the system on a specific interaction."""
        self.feedback_collector.collect_feedback(interaction_id, system_rating, system_comment)

    def analyze_performance(self):
        """Analyze the agent's performance based on collected feedback."""
        feedback_summary = self.feedback_collector.get_feedback_summary()
        # Use this summary to adjust the agent's behavior or report performance

    def analyze_and_improve(self):
        """Analyze performance and generate improvement suggestions."""
        interactions = self.interaction_logger.get_logs()
        feedback = self.feedback_collector.get_feedback()
    
        analysis_results = self.performance_analyzer.analyze_performance(interactions, feedback)
        suggestions = self.performance_analyzer.get_improvement_suggestions(analysis_results)
    
        # Use these suggestions to improve the agent's behavior
        self.implement_improvements(suggestions)
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate a response."""
        response = self.generate_response(user_input)
        predicted_satisfaction = self.performance_analyzer.predict_satisfaction(user_input, response)
    
        # You might use this prediction to adjust the response if needed
        if predicted_satisfaction < 3.0:
            response = self.generate_alternative_response(user_input)    
        return response

    def start(self) -> None:
        """Initialize the agent and start the continuous learning cycle."""
        asyncio.create_task(self.continuous_learning_cycle())
        # Other initialization code here

    def read_prompt(self, file: str, **kwargs) -> str:
        """Read a prompt template from a file."""
        content = ""
        if self.config.prompts_subdir:
            try:
                content = files.read_file(
                    files.get_abs_path(f"./prompts/{self.config.prompts_subdir}/{file}"),
                    **kwargs,
                )
            except Exception:
                pass
        if not content:
            content = files.read_file(
                files.get_abs_path(f"./prompts/default/{file}"), **kwargs
            )
        return content

    def get_data(self, field: str) -> Any:
        """Retrieve data associated with the given field from the agent's data store."""
        return self.data.get(field)

    def set_data(self, field: str, value: Any) -> None:
        """Store data associated with the given field in the agent's data store."""
        self.data[field] = value

    async def append_message(self, msg: str, human: bool = False) -> None:
        """Append a message to the agent's conversation history."""
        message_type = "human" if human else "ai"
        if self.history and self.history[-1].type == message_type:
            self.history[-1].content += "\n\n" + msg
        else:
            new_message = HumanMessage(content=msg) if human else AIMessage(content=msg)
            self.history.append(new_message)
            await self.cleanup_history(
                self.config.msgs_keep_max,
                self.config.msgs_keep_start,
                self.config.msgs_keep_end,
            )
        if message_type == "ai":
            self.last_message = msg

    def concat_messages(self, messages) -> str:
        """Concatenate a list of messages into a single string."""
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    async def send_adhoc_message(self, system: str, msg: str, output_label: str) -> str:
        """Send an ad-hoc message to the utility model and stream the response."""
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=system), HumanMessage(content=msg)]
        )

        chain = prompt | self.config.utility_model
        response = ""
        printer = None
        logger = None

        if output_label:
            PrintStyle(
                bold=True,
                font_color="orange",
                padding=True,
                background_color="white",
            ).print(f"{self.agent_name}: {output_label}:")
            printer = PrintStyle(italic=True, font_color="orange", padding=False)
            logger = self.context.log.log(
                type="adhoc", heading=f"{self.agent_name}: {output_label}:"
            )

        formatted_inputs = prompt.format()
        tokens = int(len(formatted_inputs) / 4)
        self.rate_limiter.limit_call_and_input(tokens)

        async for chunk in chain.astream({}):
            await self.handle_intervention()
            content = str(chunk.content) if hasattr(chunk, "content") else str(chunk)

            if printer:
                printer.stream(content)
            response += content
            if logger:
                logger.update(content=response)

        self.rate_limiter.set_output_tokens(int(len(response) / 4))

        return response

    def get_last_message(self) -> Optional[AIMessage | HumanMessage]:
        """Retrieve the last message from the conversation history."""
        return self.history[-1] if self.history else None

    async def replace_middle_messages(
        self, middle_messages: list[AIMessage | HumanMessage]
    ) -> list[HumanMessage]:
        """Replace a sequence of middle messages with a summary."""
        cleanup_prompt = self.read_prompt("fw.msg_cleanup.md")
        summary = await self.send_adhoc_message(
            system=cleanup_prompt,
            msg=self.concat_messages(middle_messages),
            output_label="Mid messages cleanup summary",
        )
        return [HumanMessage(content=summary)]

    async def cleanup_history(
        self, max_length: int, keep_start: int, keep_end: int
    ) -> list[AIMessage | HumanMessage]:
        """Clean up the conversation history by summarizing middle messages."""
        if len(self.history) <= max_length:
            return self.history

        first_x = self.history[:keep_start]
        last_y = self.history[-keep_end:]
        middle_part = self.history[keep_start:-keep_end]

        if middle_part and middle_part[0].type != "human":
            if first_x:
                middle_part.insert(0, first_x.pop())

        if len(middle_part) % 2 == 0:
            middle_part = middle_part[:-1]

        new_middle_part = await self.replace_middle_messages(middle_part)

        self.history = first_x + new_middle_part + last_y

        return self.history

    async def handle_intervention(self, progress: str = "") -> None:
        """Handle interventions during the message loop."""
        while self.context.paused:
            await asyncio.sleep(0.1)

        if self.intervention_message:
            msg = self.intervention_message
            self.intervention_message = ""
            if progress.strip():
                await self.append_message(progress)
            user_msg = self.read_prompt("fw.intervention.md", user_message=msg)
            await self.append_message(user_msg, human=True)
            raise InterventionException(msg)

    async def process_tools(self, msg: str) -> Optional[str]:
        """Process tool requests from the agent's message."""
        tool_request = extract_tools.json_parse_dirty(msg)

        if tool_request is not None:
            tool_name = tool_request.get("tool_name", "")
            tool_args = tool_request.get("tool_args", {})
            tool = self.get_tool(tool_name, tool_args, msg)

            await self.handle_intervention()
            await tool.before_execution(**tool_args)
            await self.handle_intervention()
            response = await tool.execute(**tool_args)
            await self.handle_intervention()
            await tool.after_execution(response)
            await self.handle_intervention()
            if response.break_loop:
                return response.message
        else:
            msg = self.read_prompt("fw.msg_misformat.md")
            await self.append_message(msg, human=True)
            PrintStyle(font_color="red", padding=True).print(msg)
            self.context.log.log(
                type="error", content=f"{self.agent_name}: Message misformat:"
            )
        return None

    def get_tool(self, name: str, args: dict, message: str, **kwargs) -> "Tool":
        """Retrieve the appropriate tool instance based on the tool name."""
        from python.tools.unknown import Unknown
        from python.helpers.tool import Tool

        tool_class = Unknown
        if files.exists("python/tools", f"{name}.py"):
            module = importlib.import_module("python.tools." + name)
            class_list = inspect.getmembers(module, inspect.isclass)

            for cls in class_list:
                if cls[1] is not Tool and issubclass(cls[1], Tool):
                    tool_class = cls[1]
                    break

        return tool_class(agent=self, name=name, args=args, message=message, **kwargs)

    async def fetch_memories(self, reset_skip: bool = False) -> str:
        """Fetch relevant memories from the agent's memory store."""
        if self.config.auto_memory_count <= 0:
            return ""
        if reset_skip:
            self.memory_skip_counter = 0

        if self.memory_skip_counter > 0:
            self.memory_skip_counter -= 1
            return ""
        else:
            self.memory_skip_counter = self.config.auto_memory_skip
            from python.tools import memory_tool

            messages = self.concat_messages(self.history)
            memories = memory_tool.search(self, messages)
            input_data = {
                "conversation_history": messages,
                "raw_memories": memories,
            }
            cleanup_prompt = self.read_prompt("msg.memory_cleanup.md").replace("{", "{{")
            clean_memories = await self.send_adhoc_message(
                cleanup_prompt, json.dumps(input_data), output_label="Memory injection"
            )
            return clean_memories

    def log_from_stream(self, stream: str, log_item: Log.LogItem) -> None:
        """Attempt to parse the stream and log key-value pairs if found."""
        try:
            if len(stream) < 25:
                return
            response = DirtyJson.parse_string(stream)
            if isinstance(response, dict):
                log_item.update(content=stream, kvps=response)
        except Exception:
            pass

    def call_extension(self, name: str, **kwargs) -> Any:
        """Placeholder for calling external extensions."""
        pass  # Implement extension calling logic here if needed

# Additional functions or classes can be added here if needed

if __name__ == "__main__":
    # Main execution code if the script is run directly
    pass