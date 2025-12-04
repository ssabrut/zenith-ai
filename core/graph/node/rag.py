from typing import Dict, Any

from core.graph.state import GraphState
from core.mcp.qdrant.server import search_knowledge_base

class RAGNode:
    async def __call__(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]

        try:
            context = await search_knowledge_base(query)
            return {"documents": context, "next_step": "generate"}
        except Exception as e:
            return {
                "documents": ["Error: Could not retrieve knowledge base."],
                "next_step": "generate"
            }