from core.graph.agent import GeneralAgent
from core.graph.state import GraphState

class GeneralNode:
    def __init__(self) -> None:
        self.agent = GeneralAgent()

    def __call__(self, state: GraphState):
        return self.agent(state)