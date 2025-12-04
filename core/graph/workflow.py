from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from core.graph.node import RouterNode, ConversationNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERATE, TOOLS
from core.globals import mcp_tools

def build_graph():
    router_node = RouterNode()
    generate_node = ConversationNode()

    workflow = StateGraph(GraphState)
    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERATE, generate_node)
    workflow.add_node(TOOLS, ToolNode(mcp_tools))

    workflow.add_edge(START, ROUTER)
    workflow.add_conditional_edges(
        ROUTER,
        lambda state: GENERATE, # For now, route everything to conversation
        {GENERATE: GENERATE}
    )

    workflow.add_conditional_edges(
        GENERATE,
        tools_condition,
        {
            TOOLS: TOOLS, 
            END: END        
        }
    )

    workflow.add_edge(TOOLS, GENERATE)
    workflow.add_edge(GENERATE, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    return app

app = build_graph()