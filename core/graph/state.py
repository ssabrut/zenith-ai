from typing import List, TypedDict, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class BookingDetails(TypedDict):
    name: Optional[str]
    phone_number: Optional[str]
    service_type: Optional[str]
    doctor: Optional[str]
    date: Optional[str]
    time: Optional[str]
    booking_id: Optional[str]

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query: str
    documents: List[str]
    booking_details: BookingDetails
    next_step: str
    tool_status: str