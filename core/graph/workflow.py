from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from core.graph.node import RouterNode, ConversationNode, RAGNode
from core.graph.state import GraphState
from core.graph.constant import ROUTER, GENERATE, RAG, BOOKING

def build_graph():
    router_node = RouterNode()
    generate_node = ConversationNode()
    rag_node = RAGNode()

    workflow = StateGraph(GraphState)
    workflow.add_node(ROUTER, router_node)
    workflow.add_node(GENERATE, generate_node)
    workflow.add_node(RAG, rag_node)

    workflow.add_edge(START, ROUTER)

    def route_decision(state: GraphState):
        step = state["next_step"]

        if step == "vectorstore":
            return RAG
        # elif step == "booking_tool":
        #     return BOOKING
        else:
            return GENERATE

    workflow.add_conditional_edges(
        ROUTER,
        route_decision,
        {
            RAG: RAG,
            # BOOKING: BOOKING,
            GENERATE: GENERATE
        }
    )

    workflow.add_edge(RAG, GENERATE)
    workflow.add_edge(GENERATE, END)
    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    return app

app = build_graph()