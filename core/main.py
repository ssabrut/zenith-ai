import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncIterator
from pydantic import ValidationError

import core.globals as global_state
from core.dependencies import load_settings
from core.config import Settings
from core.services.mcp import MCPClient
from core.routers import chat as chat_router
from core.routers import health as health_router
from core.graph.workflow import build_graph

try:
    settings: Settings = load_settings()
    MCP_SERVER_URL = "http://localhost:8001/sse"
except ValidationError as e:
    print(f"FATAL: Application configuration is invalid.\n{e}", file=sys.stderr)
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = load_settings()
    mcp_client = MCPClient(MCP_SERVER_URL)
    await mcp_client.connect()

    try:
        fetched_tools = await mcp_client.get_tools()
        global_state.mcp_tools.extend(fetched_tools)
        print(f"✅ Loaded {len(global_state.mcp_tools)} MCP tools into the graph.")

        global_state.graph_app = build_graph()
        print("✅ LangGraph built successfully with MCP tools.")
    except Exception as e:
        print(f"⚠️ Failed to load MCP tools: {e}")

    yield

    await mcp_client.disconnet()
    global_state.mcp_tools.clear()
    graph_app = None

app = FastAPI(
    title=settings.service_name,
    description="A FastAPI service for Zenith agent",
    version=settings.app_version,
    root_path="/api/v1",
    lifespan=lifespan
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": 200, "message": "ok"}

app.include_router(chat_router.router)
app.include_router(health_router.router)