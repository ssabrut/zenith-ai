from langchain_core.prompts import ChatPromptTemplate
from core.graph.state import GraphState
from core.services.deepinfra.factory import make_deepinfra_client

class GeneralAgent:
    def __init__(self):
        self.llm = make_deepinfra_client("openai/gpt-oss-20b").model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Anda adalah asisten ramah bernama Peri. Jawab sapaan dengan hangat dan singkat."),
            ("human", "{query}")
        ])

    def __call__(self, state: GraphState):
        chain = self.prompt | self.llm
        response = chain.invoke({"query": state["query"]})
        return {"messages": [response]}