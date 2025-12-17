from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import AIMessage

from core.services.deepinfra.factory import make_deepinfra_client
from core.config import get_settings
from graph.state import GraphState

class SQLAgent:
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.llm = make_deepinfra_client(model_id).model
        settings = get_settings()
        
        db_host = "db" if settings.IS_DOCKER else "localhost"
        db_uri = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{db_host}:5432/{settings.POSTGRES_DB}"
        )

        self.db = SQLDatabase.from_uri(
            db_uri,
            include_tables=["doctors", "doctor_schedules", "treatments", "appointments", "patients"]
        )

        system_prefix = """You are an expert PostgreSQL Data Analyst for a Dermatology Clinic.

        DATABASE SCHEMA MAPPING:
        - Use table `doctors` when user asks for "dokter" or "physician".
        - Use table `doctor_schedules` when user asks for "jadwal", "availability", "jam praktek", or "hari apa".
        - Use table `treatments` when user asks for "harga", "biaya", "facial", "laser", or "treatment".
        - Use table `appointments` when user asks for "booking", "janji temu", or "status".
        - Use table `patients` when user asks for "pasien" or "user data".

        IMPORTANT RULES:
        1. Always check the schema (`sql_db_schema`) for the relevant table before querying.
        2. Do NOT invent table names like "dokter" or "jadwal". Use the English names above.
        3. Use ILIKE for text search to handle case sensitivity (e.g. `name ILIKE '%budi%'`).
        4. If the user asks "Jadwal Dokter Budi", join `doctors` and `doctor_schedules`.
        """

        self.client = create_sql_agent(
            llm=self.llm,
            db=self.db,
            agent_type="tool-calling",
            verbose=True,
            prefix=system_prefix
        )

    def __call__(self, state: GraphState):
        query = state["query"]

        try:
            result = self.client.invoke({"input": query})
            print("SQLAgent result:", result)

            # Normalize the response into a string for the caller
            output = None
            if isinstance(result, dict):
                # Support common keys
                output = result.get("output") or result.get("text")

                # If no top-level text, try to extract from common generation fields
                if not output:
                    gens = result.get("generations") or result.get("choices") or result.get("data")
                    if isinstance(gens, list) and len(gens) > 0:
                        first = gens[0]
                        if isinstance(first, dict):
                            output = first.get("text") or first.get("content") or str(first)
                        else:
                            output = str(first)
            elif hasattr(result, "output"):
                output = getattr(result, "output")
            elif isinstance(result, str):
                output = result
            else:
                output = str(result)

            # Validate output isn't empty or a placeholder
            if not output or (isinstance(output, str) and ("No generation" in output or "No generation chunks" in output or output.strip() in ("", "{}", "[]"))):
                print("SQLAgent empty/invalid generation:", output)
                return {
                    "messages": [AIMessage(content="Maaf, model tidak menghasilkan respon. Silakan coba lagi nanti.")],
                    "next_step": "end"
                }

            return {"messages": [AIMessage(content=output)]}
        except Exception as e:
            err_str = str(e)
            print(f"SQLAgent error: {err_str}")

            # Handle known generation-empty errors specifically
            if "No generation" in err_str or "No generation chunks" in err_str:
                return {
                    "messages": [AIMessage(content="Maaf, model tidak menghasilkan respon (No generation chunks). Coba lagi nanti.")],
                    "next_step": "end"
                }

            return {
                "messages": [AIMessage(content="Maaf, saya tidak bisa mengakses data database saat ini.")],
                "next_step": "end"
            }