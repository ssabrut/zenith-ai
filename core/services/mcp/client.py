from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import create_model
from typing import Any
from langchain_core.tools import StructuredTool

class MCPClientManager:
    def __init__(self, url: str):
        self.url = url
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    async def connect(self):
        transport = await self.exit_stack.enter_async_context(
            sse_client(self.url)
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )

        await self.session.initialize()
        print(f"✅ Connected to MCP Server at {self.url}")

    async def disconnet(self):
        await self.exit_stack.aclose()
        print("❌ Disconnected from MCP Server")

    async def get_tools(self):
        if not self.session:
            raise RuntimeError("MCP Client not connected")

        result = await self.session.list_tools()
        tools = []

        for tool in result.tools:
            async def _tool_wrapper(**kwargs):
                return await self.session.call_tool(tool.name, arguments=kwargs)

            input_model = create_model(f"{tool.name}_input", **{
                k: (Any, ...) for k in tool.inputSchema.get("properties", {}).keys()
            })

            tools.append(StructuredTool.from_function(
                func=None,
                coroutine=_tool_wrapper,
                name=tool.name,
                description=tool.description,
                args_schema=input_model
            ))
        return tools