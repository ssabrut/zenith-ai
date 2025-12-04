from langchain_core.prompts import ChatPromptTemplate

from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client
from core.globals import mcp_tools

class InquiryNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.llm = make_deepinfra_client(model_name).model
        
        # Prompt Inquiry (Bahasa Indonesia)
        system_prompt = """Anda adalah Konsultan Estetika Senior. Tugas Anda menjawab pertanyaan tentang perawatan dan harga menggunakan alat pencarian (search tool).

        INSTRUKSI:
        1. Gunakan alat pencarian untuk menemukan fakta. Jawab HANYA berdasarkan data yang ditemukan.
        2. Gaya bicara: Informatif, to-the-point, dan membantu.
        3. Jika menyajikan harga, gunakan format daftar/tabel agar mudah dibaca.
        4. JANGAN menebak harga atau mengarang prosedur yang tidak ada di data.
        5. Akhiri dengan menawarkan bantuan lebih lanjut seputar info tersebut (bukan booking).
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

    def __call__(self, state: GraphState):
        chain = self.prompt | self.llm.bind_tools(mcp_tools)
        response = chain.invoke({"query": state["query"]})
        return {"messages": [response], "next_step": "end"}