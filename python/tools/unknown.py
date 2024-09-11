from python.helpers.tool import Tool, Response

class Unknown(Tool):
    async def execute(self, **kwargs):
        return Response(
                message=self.agent.read_prompt("fw.tool_not_found.md",
                                        tool_name=self.name,
                                        tools_prompt=self.agent.read_prompt("agent.tools.md")), 
                break_loop=False)

