from typing import List, TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    next_step: str
    booking_details: dict
    booking_active: bool