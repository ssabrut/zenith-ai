from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from core.graph.node import RouterNode, GeneralNode, InquiryNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERAL, TOOLS, INQUIRY
from core.globals import mcp_tools

def build_graph():
    router_node = RouterNode()
    general_node = GeneralNode()
    inquiry_node = InquiryNode()

    workflow = StateGraph(GraphState)
    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERAL, general_node)
    workflow.add_node(INQUIRY, inquiry_node)
    
    workflow.add_node(TOOLS, ToolNode(mcp_tools))

    workflow.add_edge(START, ROUTER)
    def route_decision(state):
        decision = state.get("next_step")
        if decision == "vectorstore":
            return INQUIRY
        else:
            return GENERAL

    workflow.add_conditional_edges(
        ROUTER,
        route_decision,
        {
            INQUIRY: INQUIRY,
            GENERAL: GENERAL
        }
    )

    workflow.add_conditional_edges(
        INQUIRY,
        tools_condition,
        {TOOLS: TOOLS, END: END}
    )

    workflow.add_edge(TOOLS, INQUIRY)
    
    workflow.add_edge(GENERAL, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app