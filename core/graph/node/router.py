from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "database_tool", "booking_tool", "general"] = Field(
        ...,
        description="Pilih rute: 'vectorstore' (info statis/harga), 'database_tool' (info real-time/jadwal), 'booking_tool' (buat janji), atau 'general' (obrolan)."
    )

class RouterNode:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        llm = make_deepinfra_client(model_name).model
        self.structured_llm = llm.with_structured_output(RouteQuery)

        system = """Anda adalah Router Cerdas untuk Klinik Dermatologi. Tugas Anda mengklasifikasikan niat pengguna ke salah satu dari 4 tujuan berikut:

        ATURAN ROUTING:
        
        1. **vectorstore**: 
           - Gunakan untuk pertanyaan UMUM dan STATIS.
           - Topik: Harga/Biaya perawatan, Penjelasan jenis facial/laser, Masalah kulit, Lokasi klinik.
           - Contoh: "Berapa harga laser?", "Apa bedanya facial A dan B?", "Klinik buka jam berapa?".
        
        2. **database_tool**: 
           - Gunakan untuk mencari data SPESIFIK/REAL-TIME (Read-Only).
           - Topik: Ketersediaan dokter, mengecek jadwal praktek, melihat status janji temu yang sudah ada.
           - Contoh: "Apakah Dr. Budi ada hari Senin?", "Siapa dokter yang tersedia besok?", "Cek jadwal Dr. Siti".
        
        3. **booking_tool**: 
           - Gunakan JIKA DAN HANYA JIKA pengguna ingin MEMBUAT, MENGUBAH, atau MEMBATALKAN janji temu (Action/Write).
           - Contoh: "Saya mau booking", "Reservasi untuk besok", "Daftarkan saya ke Dr. Siti jam 9", "Reschedule janji saya".
        
        4. **general**: 
           - Pilih ini HANYA untuk sapaan atau obrolan ringan.
           - Contoh: "Halo", "Selamat pagi", "Terima kasih", "Hai".

        PENTING:
        - Jika pengguna bertanya "Kapan Dr. Budi praktek?", arahkan ke **database_tool**.
        - Jika pengguna berkata "Saya mau ketemu Dr. Budi", arahkan ke **booking_tool**.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        query = state["query"]
        print(f"---ROUTING QUERY: {query}---")
        
        try:
            result = self.chain.invoke({"query": query})
            destination = result.datasource
        except Exception as e:
            print(f"⚠️ Routing Error: {e}. Fallback to general.")
            destination = "general"

        print(f"---ROUTED TO: {destination}---")
        return {"next_step": destination}