from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from core.services.deepinfra.factory import make_deepinfra_client
from core.schemas import BookingSchema
from core.graph.state import GraphState

class BookingAgent:
    def __init__(self, model_id: str = "openai/gpt-oss-20b"):
        self.llm = make_deepinfra_client(model_id).model
        self.extractor = self.llm.with_structured_output(BookingSchema)
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Booking Details Extractor.
            
            Your Goal: Extract ALL booking information provided by the user in the conversation.
            
            Rules:
            - If the user provides multiple details (e.g., "Name is A, Phone is B"), extract BOTH.
            - If a detail is not mentioned, return null for that field.
            - Do not guess or invent information."""),
            ("placeholder", "{messages}")
        ])

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly Clinic Receptionist.
            
            CURRENT BOOKING DATA:
            {current_details}
            
            Task:
            1. Check which fields are still 'None' (Missing).
            2. Politely ask the user for the missing information.
            3. If all fields are present, ask the user to confirm the booking details.
            
            Keep your response short and natural."""),
            ("placeholder", "{messages}")
        ])

    def __call__(self, state: GraphState):
        messages = state["messages"]
        current_details = state.get("booking_details", {})
        if not messages:
            return {"messages": [AIMessage(content="Maaf, saya tidak menangkap informasi Anda. Bisa diulangi?")]}

        current_details = state.get("booking_details", {}) or {}
        extraction_chain = self.extraction_prompt | self.extractor

        extracted_data: BookingSchema = extraction_chain.invoke({"messages": messages})
        if hasattr(current_details, "model_dump"):
            updated_details = current_details.model_dump().copy()
        else:
            updated_details = current_details.copy()

        for key, value in extracted_data.model_dump().items():
            if value is not None:
                updated_details[key] = value

        response_chain = self.response_prompt | self.llm
        response = response_chain.invoke({
            "messages": messages,
            "current_details": str(updated_details)
        })

        return {
            "booking_details": updated_details,
            "messages": [response],
            "next_step": "end"
        }