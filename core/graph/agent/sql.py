from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import AIMessage

from core.services.deepinfra.factory import make_deepinfra_client
from core.config import get_settings
from core.graph.state import GraphState

class SQLAgent:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.llm = make_deepinfra_client(model_id).model
        settings = get_settings()
        
        db_host = "db" if settings.IS_DOCKER else "localhost"
        db_uri = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{db_host}:5432/{settings.POSTGRES_DB}"
        )

        self.db = SQLDatabase.from_uri(db_uri)

        self.client = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="tool-calling",
            verbose=True,
            prefix="You are a data analyst. You must query the database to answer user questions.",
        )

    async def __call__(self, state: GraphState):
        query = state["query"]
        try:
            result = self.client.ainvoke({"input": query})
            {"messages": [AIMessage(content=result["output"])]}
        except Exception as e:
            {
                "messages": [AIMessage(content="Maaf, saya tidak bisa mengakses data database saat ini.")],
                "next_step": "end"
            }