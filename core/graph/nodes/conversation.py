from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage
from typing import Dict, Any

from core.graph.state import GraphState

class ConversationNode:
    def __init__(self, chain: Runnable) -> None:
        self.chain = chain

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        query = state["query"]
        docs = state.get("documents", [])
        formated_context = "\n\n".join(docs) if docs else ""

        try:
            response_text = self.chain.invoke({
                "query": query,
                "context": formated_context
            })

            if hasattr(response_text, "content"):
                response_text = response_text.content

            return {
                "messages": [AIMessage(content=response_text)],
                "next_step": "end"
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content="Maaf, ada kesalahan teknis saat memproses jawaban.")],
                "next_step": "end"
            }