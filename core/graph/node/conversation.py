from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from typing import Dict, Any

from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.chain import build_conversational_chain

class ConversationNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b") -> None:
        self.llm = make_deepinfra_client(model_name)
        self.chain = build_conversational_chain(self.llm)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]

        try:
            response = self.chain.invoke({"query": query})
            return {
                "messages": [AIMessage(content=response)],
                "next_step": "end"
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content="Maaf, ada kesalahan teknis saat memproses jawaban.")],
                "next_step": "end"
            }