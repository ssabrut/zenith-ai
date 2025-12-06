from langchain.agents import create_agent

from core.services.deepinfra.factory import make_deepinfra_client
from core.graph.state import GraphState
from core.graph.tools import qdrant

class InquiryAgent:
    def __init__(self, model_id: str = "openai/gpt-oss-20b"):
        self.model = make_deepinfra_client(model_id).model

        # Persona for the agent
        self.system_prompt = """Anda adalah Konsultan Estetika Senior. 
        Tugas Anda menjawab pertanyaan tentang perawatan dan harga berdasarkan data dari tools.
        
        ATURAN:
        1. Cari informasi menggunakan tools yang tersedia.
        2. Jawab HANYA berdasarkan fakta yang ditemukan.
        3. Gunakan format tabel untuk harga.
        4. Jika tidak ada info, katakan jujur.
        """

        self.agent = create_agent(
            model=self.model,
            tools=[qdrant.search_knowledge_base],
            system_prompt=self.system_prompt
        )

    async def __call__(self, state: GraphState):
        query = state["query"]
        messages = {"messages": [{"role": "user", "content": query}]}
        response = await self.agent.ainvoke(messages)
        return {"messages": [response["messages"][-1]]}