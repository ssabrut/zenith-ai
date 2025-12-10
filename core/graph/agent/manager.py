from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class Decision(BaseModel):
    next_step: Literal["inquiry", "database", "booking", "general", "FINISH"] = Field(
        ...,
        description="The worker node to call next, or `FINISH` if the AI should stop and wait for user input."
    )

class ManagerAgent:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.llm = make_deepinfra_client(model_id).model
        self.structured_llm = self.llm.with_structured_output(Decision)

        self.system_prompt = """You are the Supervisor (Manager) of a Dermatology Clinic AI.
        You manage a team of workers:
        1. **inquiry**: Answers questions about prices, treatments, and general info.
        2. **database**: Checks real-time doctor schedules and appointment status.
        3. **booking**: Handles the appointment creation process (collects Name, Date, etc.).
        4. **general**: Handles greetings and small talk.

        CONTEXT:
        Booking Active: {booking_status}

        YOUR GOAL:
        Decide the NEXT step based on the conversation history.

        RULES:
        - If the last message was a question from a worker (e.g., Booking asked "What is your name?"), output **FINISH** to wait for the user.
        - If the user asks multiple things (e.g., "Price and Book"), schedule the **inquiry** first, then the **booking**.
        - If 'Booking Active' is TRUE, prioritize **booking** UNLESS the user asks a specific question (Interruption) -> then send to **inquiry** or **database**.
        - If the user just says "Hello", send to **general**."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{messages}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        is_active = state.get("booking_active", False)

        try:
            decision = self.chain.invoke({
                "messages": state["messages"],
                "booking_status": str(is_active)
            })

            next_step = decision.next_step
        except Exception as e:
            print(f"Manager error: {e}")
            next_step = "general"

        print(f"Manager decision: {next_step}")
        return {"next_step": next_step}