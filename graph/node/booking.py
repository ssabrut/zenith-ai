from graph.agent import BookingAgent
from graph.state import GraphState

class BookingNode:
    def __init__(self):
        self.agent = BookingAgent()

    def __call__(self, state: GraphState):
        return self.agent(state)