from graph.agent import SQLAgent
from graph.state import GraphState

class SQLNode:
    def __init__(self):
        self.agent = SQLAgent()

    def __call__(self, state: GraphState):
        return self.agent(state)