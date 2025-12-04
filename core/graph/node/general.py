from langchain_core.prompts import ChatPromptTemplate

from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client

class GeneralNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.llm = make_deepinfra_client(model_name).model
        
        # Prompt General (Bahasa Indonesia)
        system_prompt = """Anda adalah Asisten Virtual yang ramah (Persona: Peri). 
        
        Tugas Anda:
        - Menjawab sapaan dengan hangat (contoh: "Halo! Ada yang bisa Peri bantu untuk kulit glowingmu hari ini?").
        - Menanggapi ucapan terima kasih dengan sopan.
        - Jika pengguna mulai bertanya teknis/medis, arahkan mereka secara halus untuk bertanya spesifik agar bisa dijawab oleh agen ahli.
        - Jaga respon tetap singkat dan ceria.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])

    def __call__(self, state: GraphState):
        chain = self.prompt | self.llm
        response = chain.invoke({"query": state["query"]})
        return {"messages": [response], "next_step": "end"}