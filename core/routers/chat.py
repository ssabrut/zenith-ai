from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage

import core.globals as global_state
from core.graph.constant import GENERAL, INQUIRY
from core.schemas import ChatRequest

router = APIRouter()

async def response_generator(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    if global_state.graph_app is None:
        yield "⚠️ System Error: The AI Graph is not initialized yet. Please check server logs."
        return
    
    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)]
    }
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for event in global_state.graph_app.astream(initial_state, config=config, version="v1"):
            for node_name, state_update in event.items():
                if node_name in [GENERAL, INQUIRY]:
                    messages = state_update.get("messages", [])
                    if messages:
                        last_msg = messages[-1]

                        yield last_msg.content
    except Exception as e:
        yield f"Error processing request: {str(e)}"

@router.post(
    "/chat",
    response_class=StreamingResponse,
    summary="Main conversational endpoint",
    tags=["chat"]
)
async def chat_endpoint(request: ChatRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return StreamingResponse(
        response_generator(request.query, request.thread_id),
        media_type="text/plain"
    )