from core.graph.agent import SQLAgent
from core.graph.state import GraphState

class SQLNode:
    def __init__(self):
        self.agent = SQLAgent()

    async def __call__(self, state: GraphState):
        return await self.agent(state)