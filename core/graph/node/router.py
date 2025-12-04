from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "booking_tool", "general_chat"] = Field(
        ...,
        description="Pilih rute: 'vectorstore' untuk info/harga/medis, 'booking_tool' untuk janji temu, atau 'general_chat' untuk obrolan ringan."
    )

class RouterNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        llm = make_deepinfra_client(model_name).model
        self.structured_llm = llm.with_structured_output(RouteQuery)

        system = """Anda adalah router ahli untuk klinik dermatologi. Tugas Anda mengarahkan pertanyaan pengguna ke agen yang tepat.

        ATURAN ROUTING:
        1. **vectorstore**: Pilih ini jika pengguna bertanya tentang:
           - Harga, biaya, atau paket perawatan.
           - Jenis facial, laser, atau produk skincare.
           - Masalah kulit (jerawat, flek, kusam).
           - Lokasi klinik atau jam operasional.

        2. **booking_tool**: Pilih ini jika pengguna ingin:
           - Membuat janji temu (booking/reservasi).
           - Mengubah jadwal (reschedule) atau membatalkan janji.
           - Mengecek jadwal dokter.

        3. **general_chat**: Pilih ini HANYA untuk:
           - Sapaan ("Halo", "Selamat pagi").
           - Ucapan terima kasih.
           - Obrolan ringan yang tidak butuh info medis atau booking.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        
        self.chain = self.prompt | self.structured_llm

    def __call__(self, state: GraphState):
        print("---ROUTING QUERY---")
        query = state["query"]
        result = self.chain.invoke({"query": query})
        print(f"---ROUTING RESULT: {result.datasource}---")
        return {"next_step": result.datasource}