from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.constant import ROUTER, GENERAL, INQUIRY, BOOKING, DATABASE, MANAGER
from graph.node import GeneralNode, InquiryNode, RouterNode, BookingNode, SQLNode, ManagerNode

def build_graph():
    workflow = StateGraph(GraphState)
    
    manager_node = ManagerNode()
    general_node = GeneralNode()
    inquiry_node = InquiryNode()
    sql_node = SQLNode()
    booking_node = BookingNode()

    workflow.add_node(MANAGER, manager_node)
    workflow.add_node(GENERAL, general_node)
    workflow.add_node(INQUIRY, inquiry_node)
    workflow.add_node(DATABASE, sql_node)
    workflow.add_node(BOOKING, booking_node)
    
    workflow.add_edge(START, MANAGER)

    def manager_routing(state: GraphState):
        step = state.get("next_step")

        if step == "FINISH":
            return END
        return step

    workflow.add_conditional_edges(
        MANAGER,
        manager_routing,
        {
            INQUIRY: INQUIRY,
            DATABASE: DATABASE,
            BOOKING: BOOKING,
            GENERAL: GENERAL,
            END: END
        }
    )

    workflow.add_edge(BOOKING, MANAGER)
    workflow.add_edge(DATABASE, END)
    workflow.add_edge(INQUIRY, END)
    workflow.add_edge(GENERAL, END)
    
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app