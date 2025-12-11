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

        KONTEKS SAAT INI:
        Booking Active: {booking_status}
        (TRUE = User sedang dalam proses mengisi form reservasi).

        TUGAS ANDA:
        Arahkan pesan user ke agen yang tepat.

        ATURAN ROUTING (PRIORITAS TINGGI):

        1. **ATURAN KHUSUS 'BOOKING ACTIVE' = TRUE**:
           - JIKA user memberikan **DATA** (Contoh: "Nama saya Budi", "0812345", "Senin depan", "Jam 10", "Ya/Tidak") -> **WAJIB ke 'booking'**.
             (Ini berarti user sedang melanjutkan pengisian form yang tertunda).
           - JIKA user bertanya info ("Berapa harganya?", "Apa itu facial?") -> Ke **'inquiry'** (Interupsi).
           - JIKA user bertanya jadwal ("Dokter Budi ada?") -> Ke **'database'** (Interupsi).

        2. **ATURAN UMUM**:
           - Ingin reservasi/janji temu -> **'booking'**.
           - Pertanyaan medis/harga/lokasi -> **'inquiry'**.
           - Cek jadwal/ketersediaan dokter -> **'database'**.
           - Sapaan ("Halo", "Pagi") -> **'general'**.

        3. **STOP CONDITION**:
           - Jika pesan TERAKHIR adalah pertanyaan dari AI (misal: "Siapa nama Anda?"), outputkan **FINISH**.
           - Jangan panggil agen dua kali untuk hal yang sama."""

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