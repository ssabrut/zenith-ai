from graph.agent import InquiryAgent
from graph.state import GraphState
class InquiryNode:
    def __init__(self) -> None:
        self.agent = InquiryAgent()

    async def __call__(self, state: GraphState):
        return await self.agent(state)