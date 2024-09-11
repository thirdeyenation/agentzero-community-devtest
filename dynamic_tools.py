import importlib
import inspect
from typing import Dict, Any, Callable

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, func: Callable):
        self.tools[name] = func

    def get_tool(self, name: str) -> Callable:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

class DynamicToolManager:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    def create_tool(self, tool_spec: Dict[str, Any]) -> None:
        tool_name = tool_spec['name']
        tool_code = tool_spec['code']

        # Create a new module for the tool
        module = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(tool_name, loader=None)
        )

        # Execute the tool code in the module's context
        exec(tool_code, module.__dict__)

        # Find the main function in the module
        main_func = None
        for name, obj in module.__dict__.items():
            if inspect.isfunction(obj) and name.startswith('tool_'):
                main_func = obj
                break

        if main_func is None:
            raise ValueError("No tool function found in the provided code.")

        # Register the tool
        self.tool_registry.register_tool(tool_name, main_func)

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        return tool(**kwargs)

# Usage
tool_registry = ToolRegistry()
dynamic_tool_manager = DynamicToolManager(tool_registry)

# Example: Dynamically create a new tool
new_tool_spec = {
    "name": "text_summarizer",
    "code": """
def tool_summarize(text: str, max_length: int = 100) -> str:
    # Simple summarization logic (replace with more sophisticated algorithm)
    words = text.split()
    if len(words) <= max_length:
        return text
    return ' '.join(words[:max_length]) + '...'
    """
}

dynamic_tool_manager.create_tool(new_tool_spec)

# Execute the dynamically created tool
result = dynamic_tool_manager.execute_tool("text_summarizer", 
                                           text="This is a long text that needs summarization...", 
                                           max_length=10)
print(result)