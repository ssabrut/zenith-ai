from langchain_core.messages import AIMessage
from typing import Dict, Any

from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.chain import build_conversational_chain
from core.globals import mcp_tools

class GeneralNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b") -> None:
        self.llm = make_deepinfra_client(model_name)
        self.chain = build_conversational_chain(self.llm.model)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]

        if mcp_tools:
            chain = self.chain.bind_tools(mcp_tools)
        else:
            chain = self.chain

        try:
            response = chain.invoke({"query": query})
            return {
                "messages": [response],
                "next_step": "decide"
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content="Maaf, ada kesalahan teknis saat memproses jawaban.")],
                "next_step": "end"
            }