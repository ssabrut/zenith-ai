from mcp import ClientSession
from typing import Dict, Any
from mcp.client.sse import sse_client

from core.graph.state import GraphState

MCP_SERVER_URL = f"http://localhost:8000/tools/mcp"

class RAGNode:
    async def __call__(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]

        try:
            async with sse_client(MCP_SERVER_URL) as streams:
                async with ClientSession(streams.read, streams.write) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        "search_knowledge_base",
                        arguments={"query": query}
                    )

                    return {
                        "documents": result,
                        "next_step": "generate" 
                    }
        except Exception as e:
            return {
                "documents": ["Error: Could not retrieve knowledge base."],
                "next_step": "generate"
            }