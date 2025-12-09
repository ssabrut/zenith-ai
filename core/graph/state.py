from typing import List, TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from core.schemas import BookingSchema

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    documents: List[str]
    next_step: str
    tool_status: str
    booking_details: BookingSchema