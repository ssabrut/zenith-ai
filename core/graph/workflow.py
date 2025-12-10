from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERAL, INQUIRY, BOOKING, DATABASE
from core.graph.node import GeneralNode, InquiryNode, RouterNode, BookingNode, SQLNode

def build_graph():
    workflow = StateGraph(GraphState)
    
    router_node = RouterNode()
    general_node = GeneralNode()
    inquiry_node = InquiryNode()
    sql_node = SQLNode()
    booking_node = BookingNode()

    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERAL, general_node)
    workflow.add_node(INQUIRY, inquiry_node)
    workflow.add_node(DATABASE, sql_node)
    workflow.add_node(BOOKING, booking_node)
    
    workflow.add_edge(START, ROUTER)
    
    def route_decision(state):
        decision = state.get("next_step")
        if decision == "vectorstore":
            return INQUIRY
        elif decision == "database_tool":
            return DATABASE
        elif decision == "booking_tool":
            return BOOKING
        return GENERAL

    workflow.add_conditional_edges(
        ROUTER,
        route_decision,
        {
            INQUIRY: INQUIRY,
            DATABASE: DATABASE,
            GENERAL: GENERAL,
            BOOKING: BOOKING
        }
    )

    workflow.add_edge(INQUIRY, END)
    workflow.add_edge(GENERAL, END)
    workflow.add_edge(DATABASE, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app