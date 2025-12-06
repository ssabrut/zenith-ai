from langchain.agents import create_agent
from core.services.deepinfra.factory import make_deepinfra_client

def build_inquiry_agent(tools: list):
    """
    Builds a ReAct agent that handles the Inquiry loop (Search -> Answer).
    """
    model = make_deepinfra_client("openai/gpt-oss-20b").model

    # Persona for the agent
    system_prompt = """Anda adalah Konsultan Estetika Senior. 
    Tugas Anda menjawab pertanyaan tentang perawatan dan harga berdasarkan data dari tools.
    
    ATURAN:
    1. Cari informasi menggunakan tools yang tersedia.
    2. Jawab HANYA berdasarkan fakta yang ditemukan.
    3. Gunakan format tabel untuk harga.
    4. Jika tidak ada info, katakan jujur.
    """

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent