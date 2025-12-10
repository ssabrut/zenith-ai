from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class RouteQuery(BaseModel):
    datasource: Literal["inquiry", "database", "booking", "general"] = Field(
        ...,
        description="Pilih rute: 'inquiry' (info statis/harga), 'database' (info real-time/jadwal), 'booking' (buat janji), atau 'general' (obrolan)."
    )

class RouterNode:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        llm = make_deepinfra_client(model_name).model
        self.structured_llm = llm.with_structured_output(RouteQuery)

        system = """You are the Main Dispatcher for a Dermatology Clinic.
        
        CONTEXT:
        Booking In-Progress: {booking_status} 
        (If TRUE, the user is currently in the middle of a booking flow).

        ROUTING RULES:
        
        1. **booking**:
           - PRIORITIZE this if 'Booking In-Progress' is TRUE and the user provides data (Name, Date, "Yes", "No").
           - Use this if the user EXPLICITLY wants to start booking ("I want to book", "Make appointment").
           - Use this if the user wants to CANCEL ("Cancel booking").
           
        2. **inquiry** (Inquiry):
           - Use this for questions about Prices, Treatments, Locations, or Medical Info.
           - EVEN IF 'Booking In-Progress' is TRUE, if the user asks a question (e.g., "Wait, how much is it?"), route here. (This is an INTERRUPTION).

        3. **database**:
           - Use for checking Doctor Schedules or specific Availability ("Is Dr. Budi available?").
           - This is also an INTERRUPTION if booking is in progress.

        4. **general**:
           - Only for "Hi", "Thanks", "Bye"."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        query = state["query"]
        is_active = state.get("booking_active", False)
        status_str = "True" if is_active else "False"
        
        print(f"---ROUTING QUERY: {query}---")
        print(f"---ROUTING (BookingActive: {status_str})---")
        
        try:
            result = self.chain.invoke({
                "query": query, 
                "booking_status": status_str
            })
            destination = result.datasource
        except Exception:
            destination = "general"

        print(f"---ROUTED TO: {destination}---")
        return {"next_step": destination}