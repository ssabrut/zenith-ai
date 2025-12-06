from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from loguru import logger

from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client
from core.globals import mcp_tools

class InquiryNode:
    """
    Handles the inquiry processing logic within the conversation graph.

    This node utilizes a DeepInfra LLM agent bound with MCP tools to answer
    user queries regarding aesthetic treatments and pricing. It enforces
    strict fact-based responses derived solely from the provided search tools.

    Attributes:
        llm (BaseChatModel): The initialized language model client.
        prompt (ChatPromptTemplate): The constructed system and human prompt template.
    """

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """
        Initializes the InquiryNode with a specific model and system prompt.

        Args:
            model_name (str): The identifier for the DeepInfra model to be used.
                              Defaults to "openai/gpt-oss-20b".

        Raises:
            ValueError: If the model client cannot be initialized.
            RuntimeError: If mcp_tools are not correctly loaded in global scope.
        """
        try:
            self.llm = make_deepinfra_client(model_name).model
        except Exception as e:
            logger.critical(f"Failed to initialize LLM client for model '{model_name}': {e}")
            raise ValueError(f"Could not initialize InquiryNode due to LLM failure: {e}")

        if not mcp_tools:
            logger.warning("mcp_tools is empty or None. The agent will have no tools to use.")

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
        
        logger.info(f"InquiryNode initialized with model: {model_name}")

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Executes the inquiry logic against the current state.

        Constructs the chain by binding the LLM with tools and invoking it
        with the user's query.

        Args:
            state (GraphState): The current state of the conversation graph, 
                                expecting a 'query' key.

        Returns:
            Dict[str, Any]: A dictionary containing the generated 'messages' 
                            and the 'next_step' indicator.

        Raises:
            KeyError: If 'query' is missing from the state.
        """
        query = state.get("query")
        
        if not query:
            logger.error("State is missing 'query'. Cannot proceed with inquiry.")
            return {
                "messages": [AIMessage(content="Maaf, saya tidak menerima pertanyaan yang valid.")],
                "next_step": "end"
            }

        logger.info(f"Processing inquiry for query: {query}...")

        try:
            # Bind tools to the LLM for this specific call
            chain = self.prompt | self.llm.bind_tools(mcp_tools)
            response = chain.invoke({"query": query})

            return {"messages": [response], "next_step": "end"}
        except Exception as e:
            logger.exception("Error occurred during InquiryNode execution.")
            # Fail gracefully by returning an error message to the user rather than crashing the graph
            error_message = AIMessage(content="Maaf, terjadi kesalahan teknis saat memproses pertanyaan Anda. Silakan coba lagi nanti.")
            return {"messages": [error_message], "next_step": "end"}