from core.graph.agent import InquiryAgent
from core.graph.state import GraphState
class InquiryNode:
    def __init__(self) -> None:
        self.agent = InquiryAgent()

    async def __call__(self, state: GraphState):
        return await self.agent(state)