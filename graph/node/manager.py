from graph.agent import ManagerAgent
from graph.state import GraphState

class ManagerNode:
    def __init__(self):
        self.agent = ManagerAgent()

    def __call__(self, state: GraphState):
        return self.agent(state)