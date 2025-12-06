from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from core.graph.node import RouterNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERAL, INQUIRY
from core.graph.agent import GeneralAgent, InquiryAgent

def build_graph():
    workflow = StateGraph(GraphState)
    
    router_node = RouterNode()
    general_agent = GeneralAgent()
    inquiry_agent = InquiryAgent()

    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERAL, general_agent)
    workflow.add_node(INQUIRY, inquiry_agent)
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