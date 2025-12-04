from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from core.graph.node import RouterNode, GeneralNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERAL, TOOLS
from core.globals import mcp_tools

def build_graph():
    router_node = RouterNode()
    general_node = GeneralNode()

    workflow = StateGraph(GraphState)
    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERAL, general_node)
    workflow.add_node(TOOLS, ToolNode(mcp_tools))

    workflow.add_edge(START, ROUTER)
    workflow.add_conditional_edges(
        ROUTER,
        lambda state: GENERAL, # For now, route everything to conversation
        {GENERAL: GENERAL}
    )

    workflow.add_conditional_edges(
        GENERAL,
        tools_condition,
        {
            TOOLS: TOOLS, 
            END: END        
        }
    )

    workflow.add_edge(TOOLS, GENERAL)
    workflow.add_edge(GENERAL, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app

app = build_graph()