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

        system = """Anda adalah Router Cerdas.
        
        STATUS BOOKING SAAT INI: {booking_status}
        (Jika 'True', berarti user sedang dalam proses tanya jawab data booking)

        ATURAN ROUTING:
        
        KONDISI 1: JIKA BOOKING STATUS = 'True':
        - Prioritaskan **booking_tool** jika input user berupa jawaban pendek (Nama, Tanggal, Jam, Ya/Tidak).
        - Hanya pilih **vectorstore** jika user bertanya soal harga/produk ("Berapa harganya?", "Apa itu facial A?").
        - Hanya pilih **database_tool** jika user bertanya jadwal dokter ("Dr Budi ada?").
        - Pilih **booking_tool** untuk melanjutkan proses data.

        KONDISI 2: JIKA BOOKING STATUS = 'False':
        - **vectorstore**: Pertanyaan umum/harga/medis.
        - **database_tool**: Cek jadwal/status real-time.
        - **booking_tool**: Ingin membuat janji baru.
        - **general**: Sapaan/obrolan ringan.
        """

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