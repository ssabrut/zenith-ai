from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "booking_tool", "general_chat"] = Field(
        ...,
        description="Given a user query, choost to route it to `vectorstore` (for information/prices/inquiry), `booking_tool` (for appointments), or `general_chat`."
    )

class RouterNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        llm = make_deepinfra_client(model_name).model
        self.structured_llm = llm.with_structured_output(RouteQuery)

        system = """You are an expert router.
        - If the user asks about treatments, prices, locations, or skin problems, route to 'vectorstore'.
        - If the user wants to book, cancel, reschedule, or check appointments, route to 'booking_tool'.
        - Otherwise, route to 'general_chat'."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        print("---ROUTING QUERY---")
        query = state["query"]
        result = self.chain.invoke({"query": query})
        return {"next_step": result.datasource}