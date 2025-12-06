from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from core.graph.node import RouterNode, GeneralNode, InquiryNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERAL, TOOLS, INQUIRY
from core.graph.agent.inquiry import build_inquiry_agent
from core.graph.agent.general import GeneralAgent

def build_graph():
    workflow = StateGraph(GraphState)
    
    router_node = RouterNode()
    general_agent = GeneralAgent()
    inquiry_subgraph = build_inquiry_agent()

    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERAL, general_agent)

    async def call_inquiry(state: GraphState):
        query = state["query"]
        messages = {"messages": [{"role": "user", "content": query}]}
        response = await inquiry_subgraph.ainvoke(messages)
        return {"messages": [response["messages"][-1]]}
        
    workflow.add_node(INQUIRY, call_inquiry)
    workflow.add_edge(START, ROUTER)
    
    def route_decision(state):
        decision = state.get("next_step")
        if decision == "vectorstore":
            return INQUIRY
        return GENERAL

    workflow.add_conditional_edges(
        ROUTER,
        route_decision,
        {
            INQUIRY: INQUIRY,
            GENERAL: GENERAL
        }
    )

    workflow.add_edge(INQUIRY, END)
    workflow.add_edge(GENERAL, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app