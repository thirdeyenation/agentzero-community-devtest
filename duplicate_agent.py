import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional
import uuid

from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings

import python.helpers.log as Log
from python.helpers.dirty_json import DirtyJson
from python.helpers.defer import DeferredTask
from python.helpers import extract_tools, rate_limiter, files, errors
from python.helpers.print_style import PrintStyle

# Continuous Learning Pipeline components
from agent_zero.continuous_learning.data_collector import DataCollector
from agent_zero.continuous_learning.knowledge_integrator import KnowledgeIntegrator
from agent_zero.continuous_learning.interaction_logger import InteractionLogger
from agent_zero.continuous_learning.feedback_collector import FeedbackCollector
from agent_zero.continuous_learning.performance_analyzer import PerformanceAnalyzer
from agent_zero.continuous_learning.content_filter import ContentFilter
from agent_zero.continuous_learning.human_verification import HumanVerification
from agent_zero.continuous_learning.bias_detector import BiasDetector
from agent_zero.knowledge_base import KnowledgeBase
from agent_zero.vector_db import VectorDB


class AgentContext:
    """
    Manages the context for Agent Zero instances, including configuration, logging, 
    and communication with the main agent.
    """

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
        return AgentContext._contexts.get(id)

    @staticmethod
    def first() -> Optional["AgentContext"]:
        if not AgentContext._contexts:
            return None
        return list(AgentContext._contexts.values())[0]

    @staticmethod
    def remove(id: str) -> Optional["AgentContext"]:
        context = AgentContext._contexts.pop(id, None)
        if context and context.process:
            context.process.kill()
        return context

    def reset(self):
        if self.process:
            self.process.kill()
        self.log.reset()
        self.agent0 = Agent(0, self.config, self)
        self.streaming_agent = None
        self.paused = False

    def communicate(self, msg: str, broadcast_level: int = 1) -> DeferredTask:
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
    """
    Stores the configuration settings for an Agent Zero instance.
    """
    chat_model: BaseChatModel | BaseLLM
    utility_model: BaseChatModel | BaseLLM
    embeddings_model:Embeddings
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
    code_exec_docker_ports: dict[str,int] = field(default_factory=lambda: {"22/tcp": 50022})
    code_exec_docker_volumes: dict[str, dict[str, str]] = field(default_factory=lambda: {files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"}})
    code_exec_ssh_enabled: bool = True
    code_exec_ssh_addr: str = "localhost"
    code_exec_ssh_port: int = 50022
    code_exec_ssh_user: str = "root"
    code_exec_ssh_pass: str = "toor"
    additional: Dict[str, Any] = field(default_factory=dict)

class InterventionException(Exception):
    """
    An exception raised to signal an intervention during the agent's message loop.
    This allows the loop to be interrupted and resumed later.
    """

    pass

class KillerException(Exception):
    """
    A critical exception that terminates the agent's message loop and cannot be 
    handled by the agent itself.
    """

    pass

class Agent:
    """
    The core class representing an Agent Zero instance. It handles message processing,
    tool execution, memory management, and continuous learning.
    """

    def __init__(
        self, number: int, config: AgentConfig, context: Optional[AgentContext] = None
    ):
        self.config = config
        self.context = context or AgentContext(config)
        self.number = number
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
        self.knowledge_integrator = KnowledgeIntegrator(
            self.knowledge_base, self.vector_db
        )
        self.interaction_logger = InteractionLogger()
        self.feedback_collector = FeedbackCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.content_filter = ContentFilter()
        self.human_verification = HumanVerification()
        self.bias_detector = BiasDetector()


    async def message_loop(self, msg: str):
        """
        The main message processing loop for the agent. It handles user input,
        generates responses, manages tool execution, and handles interventions.
        """
        try:
            printer = PrintStyle(italic=True, font_color="#b3ffd9", padding=False)    
            user_message = self.read_prompt("fw.user_message.md", message=msg)
            await self.append_message(user_message, human=True) # Append the user's input to the history                        
            memories = await self.fetch_memories(True)
                
            while True: # let the agent iterate on his thoughts until he stops by using a tool
                self.context.streaming_agent = self #mark self as current streamer
                agent_response = ""

                try:
                    system = (
                        self.read_prompt("agent.system.md", agent_name=self.agent_name)
                        + "\n\n"
                        + self.read_prompt("agent.tools.md")
                    )
                    memories = await self.fetch_memories()
                    if memories:
                        system += "\n\n" + memories

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            SystemMessage(content=system),
                            MessagesPlaceholder(variable_name="messages"),
                        ]
                    )

                    inputs = {"messages": self.history}
                    chain = prompt | self.config.chat_model

                    formatted_inputs = prompt.format(messages=self.history)
                    tokens = int(len(formatted_inputs) / 4)
                    self.rate_limiter.limit_call_and_input(tokens)
                    
                                      # Output that the agent is starting
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
                        await self.handle_intervention(
                            agent_response
                        )  # Wait for intervention

                        if isinstance(chunk, str):
                            content = chunk
                        elif hasattr(chunk, "content"):
                            content = str(chunk.content)
                        else:
                            content = str(chunk)
                        
                        if content:
                            printer.stream(content) # output the agent response stream
                            agent_response += content # concatenate stream into the response
                            self.log_from_stream(agent_response, log)

                    self.rate_limiter.set_output_tokens(int(len(agent_response) / 4)) # rough estimation
                    
                    await self.handle_intervention(agent_response)

                    if self.last_message == agent_response: #if assistant_response is the same as last message in history, let him know
                        await self.append_message(agent_response) # Append the assistant's response to the history
                        warning_msg = self.read_prompt("fw.msg_repeat.md")
                        await self.append_message(warning_msg, human=True) # Append warning message to the history
                        PrintStyle(font_color="orange", padding=True).print(warning_msg)
                        self.context.log.log(type="warning", content=warning_msg)
                    else: #otherwise proceed with tool
                        await self.append_message(agent_response) # Append the assistant's response to the history
                        tools_result = await self.process_tools(agent_response) # process tools requested in agent message
                        if tools_result: #final response of message loop available
                            return tools_result #break the execution if the task is done

                 except InterventionException:
                    pass  # Intervention message handled in handle_intervention()
                except asyncio.CancelledError as e:
                   PrintStyle(
                        font_color="white", background_color="red", padding=True
                    ).print(f"Context {self.context.id} terminated during message loop")
                    raise e# process cancelled from outside, kill the loop
                except KillerException as e:
                    error_message = errors.format_error(e)
                    self.context.log.log(type="error", content=error_message)
                    raise e# kill the loop
                except Exception as e: # Forward other errors to the LLM, maybe it can fix them
                    error_message = errors.format_error(e)
                    msg_response = self.read_prompt("fw.error.md", error=error_message) # error message template
                    await self.append_message(msg_response, human=True)
                    PrintStyle(font_color="red", padding=True).print(msg_response)
                    self.context.log.log(type="error", content=msg_response)
                    
        finally:
            self.context.streaming_agent = None # unset current streamer

    async def continuous_learning_cycle(self):
        """
        The continuous learning cycle for the agent. It collects new data, filters
        and verifies it, integrates it into the knowledge base, analyzes performance,
        adapts the model, checks for bias, and creates snapshots periodically.
        """

        while True:
            # Collect new data
            new_data = await self.data_collector.collect()

            # Filter and verify content
            filtered_data = [
                d
                for d in new_data
                if not self.content_filter.filter_content(d["content"])
            ]
            for item in filtered_data:
                self.human_verification.add_for_review(item["content"])
            verified_data = self.human_verification.review_content()
            
           # Integrate new knowledge
            self.knowledge_integrator.integrate_new_knowledge(verified_data)

            # Analyze performance and adapt
            interactions = self.interaction_logger.get_logs()
            feedback = self.feedback_collector.get_feedback()
            self.performance_analyzer.train_model(interactions, feedback)

            # Check for bias
            recent_predictions = [
                self.performance_analyzer.predict_satisfaction(i)
                for i in interactions[-100:]
            ]
            sensitive_features = [
                i.split("|")[0] for i in interactions[-100:]
            ]  # Assuming first part of log is a sensitive feature
            if not self.bias_detector.check_bias(
                recent_predictions, sensitive_features
            ):
                print("Bias detected. Initiating mitigation strategies.")
                # Implement bias mitigation strategies here

            # Create snapshots
            self.knowledge_integrator.create_snapshot()

            # Wait for next cycle
            await asyncio.sleep(86400)  # Run daily

    def process_input(self, user_input: str) -> str:
        """
        Processes user input, triggers the message loop, logs the interaction, 
        and returns the agent's response.
        """

        # Existing input processing logic here (if any)
        response = asyncio.run(self.message_loop(user_input)) 
        self.interaction_logger.log_interaction(user_input, response)
        return response

    def collect_feedback(
        self, interaction_id: str, rating: int, comment: str = None
    ) -> None:
        """
        Collects feedback on a specific interaction.
        """

        self.feedback_collector.collect_feedback(interaction_id, rating, comment)

    def start(self) -> None:
        """
        Initializes the agent and starts the continuous learning cycle.
        """

        asyncio.create_task(self.continuous_learning_cycle())
        # Other initialization code here

    # ... [Keep other existing methods] ...

    def read_prompt(self, file: str, **kwargs) -> str:
        """
        Reads a prompt template from a file, prioritizing the prompts_subdir if specified.
        """

        content = ""
        if self.config.prompts_subdir:
            try:
                content = files.read_file(
                    files.get_abs_path(
                        f"./prompts/{self.config.prompts_subdir}/{file}"
                    ),
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
        """
        Retrieves data associated with the given field from the agent's data store.
        """

        return self.data.get(field)

    def set_data(self, field: str, value: Any) -> None:
        """
        Stores data associated with the given field in the agent's data store.
        """

        self.data[field] = value

    async def append_message(self, msg: str, human: bool = False) -> None:
        """
        Appends a message to the agent's conversation history and cleans up the 
        history if it exceeds the maximum length.
        """

        message_type = "human" if human else "ai"
        if self.history and self.history[-1].type == message_type:
            self.history[-1].content += "\n\n" + msg
        else:
            new_message = (
                HumanMessage(content=msg) if human else AIMessage(content=msg)
            )
            self.history.append(new_message)
            await self.cleanup_history(
                self.config.msgs_keep_max,
                self.config.msgs_keep_start,
                self.config.msgs_keep_end,
            )
        if message_type == "ai":
            self.last_message = msg

    def concat_messages(self, messages) -> str:
        """
        Concatenates a list of messages into a single string.
        """

        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    async def send_adhoc_message(
        self, system: str, msg: str, output_label: str
    ) -> str:
        """
        Sends an ad-hoc message to the utility model and streams the response.
        """

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
            if self.handle_intervention():
                break # wait for intervention and handle it, if paused

                        if isinstance(chunk, str):
                            content = chunk
                        elif hasattr(chunk, "content"):
                            content = str(chunk.content)
                        else:
                            content = str(chunk)

                        if printer:
                            printer.stream(content)
                        response += content
                        if logger:
                            logger.update(content=response)

        self.rate_limiter.set_output_tokens(int(len(response) / 4))

        return response
            
    def get_last_message(self) -> Optional[AIMessage | HumanMessage]:
        """
        Retrieves the last message from the conversation history.
        """

        if self.history:
            return self.history[-1]
        return None  # Explicitly return None if history is empty

    async def replace_middle_messages(
        self, middle_messages: list[AIMessage | HumanMessage]
    ) -> list[HumanMessage]:
        """
        Replaces a sequence of middle messages in the conversation history with 
        a summary generated by the utility model.
        """

        cleanup_prompt = self.read_prompt("fw.msg_cleanup.md")
        summary = await self.send_adhoc_message(
            system=cleanup_prompt,
            msg=self.concat_messages(middle_messages),
            output_label="Mid messages cleanup summary",
        )
        new_human_message = HumanMessage(content=summary)
        return [new_human_message]

    async def cleanup_history(
        self, max_length: int, keep_start: int, keep_end: int
    ) -> list[AIMessage | HumanMessage]:
        """
        Cleans up the conversation history by summarizing middle messages if the
        history exceeds the maximum length.
        """

        if len(self.history) <= max_length:
            return self.history

        first_x = self.history[:keep_start]
        last_y = self.history[-keep_end:]

        # Identify the middle part
        middle_part = self.history[keep_start:-keep_end]

        # Ensure the first message in the middle is "human", if not, move one message back
        if middle_part and middle_part[0].type != "human":
            if len(first_x) > 0:
                middle_part.insert(0, first_x.pop())

        # Ensure the middle part has an odd number of messages
        if len(middle_part) % 2 == 0:
            middle_part = middle_part[:-1]

        # Replace the middle part using the replacement function
        new_middle_part = await self.replace_middle_messages(middle_part)

        self.history = first_x + new_middle_part + last_y

        return self.history

    async def handle_intervention(self, progress: str = "") -> None:
        """
        Handles interventions during the message loop, appending the current progress 
        and the intervention message to the history.
        """

        while self.context.paused:
            await asyncio.sleep(0.1)

        if self.intervention_message:
            msg = self.intervention_message
            self.intervention_message = ""
            if progress.strip():
                await self.append_message(progress)
            user_msg = self.read_prompt(
                "fw.intervention.md", user_message=msg
            )
            await self.append_message(user_msg, human=True)
            raise InterventionException(msg)

    async def process_tools(self, msg: str) -> Optional[str]:
        """
        Processes tool requests from the agent's message, executes the requested tool, 
        and handles interventions during execution.
        """

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
        return None  # Explicitly return None if no tool result

     def get_tool(
        self, name: str, args: dict, message: str, **kwargs
    ) -> "Tool":  # Assuming 'Tool' is defined elsewhere
        """
        Retrieves the appropriate tool instance based on the tool name.
        """

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
        """
        Fetches relevant memories from the agent's memory store based on the 
        conversation history.
        """

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
            cleanup_prompt = self.read_prompt("msg.memory_cleanup.md").replace(
                "{", "{{"
            )
            clean_memories = await self.send_adhoc_message(
                cleanup_prompt, json.dumps(input_data), output_label="Memory injection"
            )
            return clean_memories

    def log_from_stream(self, stream: str, log_item: Log.LogItem) -> None:
        """
        Attempts to parse the stream and log key-value pairs if found.
        """

        try:
            if len(stream) < 25:
                return
            response = DirtyJson.parse_string(stream)
            if isinstance(response, dict):
                log_item.update(content=stream, kvps=response)
        except Exception:
            pass   

    def call_extension(self, name: str, **kwargs) -> Any:
        """
        Placeholder for calling external extensions.
        """
        pass  # Implement extension calling logic here if needed

#------------------------------------#
#   Continuous Learning Pipeline     #
#------------------------------------#
    async def continuous_learning_cycle(self) -> None:
        """
        The continuous learning cycle for the agent. It collects new data, filters
        and verifies it, integrates it into the knowledge base, analyzes performance,
        adapts the model, checks for bias, and creates snapshots periodically.
        """

        while True:
            # 1. Collect new data
            new_data = await self.data_collector.collect()

            # 2. Filter and verify content
            filtered_data = [
                d for d in new_data if not self.content_filter.filter_content(d["content"])
            ]
            for item in filtered_data:
                self.human_verification.add_for_review(item["content"])
            verified_data = self.human_verification.review_content()

            # 3. Integrate new knowledge
            self.knowledge_integrator.integrate_new_knowledge(verified_data)

            # 4. Analyze performance and adapt
            interactions = self.interaction_logger.get_logs()
            feedback = self.feedback_collector.get_feedback()
            self.performance_analyzer.train_model(interactions, feedback)

            # 5. Check for bias
            recent_predictions = [
                self.performance_analyzer.predict_satisfaction(i) for i in interactions[-100:]
            ]
            sensitive_features = [i.split("|")[0] for i in interactions[-100:]]  # Assuming first part of log is a sensitive feature
            if not self.bias_detector.check_bias(recent_predictions, sensitive_features):
                print("Bias detected. Initiating mitigation strategies.")
                # TODO: Implement bias mitigation strategies here

            # 6. Create snapshots
            self.knowledge_integrator.create_snapshot()

            # 7. Wait for next cycle
            await asyncio.sleep(86400)  # Run daily
          