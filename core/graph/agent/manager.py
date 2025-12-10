from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class Decision(BaseModel):
    next_step: Literal["inquiry", "database", "booking", "general", "FINISH"] = Field(
        ...,
        description="Langkah selanjutnya atau 'FINISH'."
    )

class ManagerAgent:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.llm = make_deepinfra_client(model_id).model
        self.structured_llm = self.llm.with_structured_output(Decision)

        self.system_prompt = """Anda adalah Supervisor (Manager) AI Klinik.
        
        TUGAS ANDA:
        Memutuskan siapa yang berbicara selanjutnya berdasarkan pesan TERAKHIR.

        PEKERJA:
        - inquiry, database, booking, general.
        - FINISH: Gunakan ini untuk berhenti dan menunggu input user.

        ATURAN PENTING UNTUK MENCEGAH LOOP:
        1. **CEK PESAN TERAKHIR**:
           - Jika pesan terakhir adalah dari **AI (Assistant)**: Outputkan **FINISH**.
           - (Kecuali jika Anda perlu memanggil agen lain secara berantai, misal: User tanya harga DAN booking).
        
        2. **JANGAN ULANGI**:
           - Jika user berkata "Halo" dan AI sudah menjawab "Halo", JANGAN panggil 'general' lagi. Outputkan **FINISH**.

        3. **CHAINING**:
           - Jika User minta "Cek harga lalu booking", dan 'inquiry' baru saja menjawab harga -> Panggil 'booking'.
           - Jika 'booking' baru saja bertanya "Siapa nama Anda?" -> Outputkan **FINISH**."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{messages}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        is_active = state.get("booking_active", False)
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        if isinstance(last_message, AIMessage):
            print("🔍 Pesan terakhir dari AI. Memeriksa apakah perlu lanjut...")

        try:
            decision = self.chain.invoke({
                "messages": messages,
                "booking_status": str(is_active)
            })

            if decision is None:
                print("⚠️ Manager returned None. Fallback to 'general'.")
                next_step = "general"
            else:
                next_step = decision.next_step
        except Exception as e:
            print(f"Manager error: {e}")
            next_step = "general"

        print(f"Manager decision: {next_step}")
        return {"next_step": next_step}