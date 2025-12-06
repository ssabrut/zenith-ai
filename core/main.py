import asyncio
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
    MCP_SERVER_URL = settings.MCP_SERVER_URL
except ValidationError as e:
    print(f"FATAL: Application configuration is invalid.\n{e}", file=sys.stderr)
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = load_settings()
    mcp_client = MCPClient(MCP_SERVER_URL)

    print(f"🔌 Connecting to MCP Server at {MCP_SERVER_URL}...")
    connected = False
    for attempt in range(15):
        try:
            await mcp_client.connect()
            connected = True
            print("✅ Successfully connected to MCP Server.")
            break
        except Exception as e:
            # It is normal for this to fail a few times while the other container starts
            print(f"⏳ MCP Server not ready yet. Retrying in 2s... (Attempt {attempt+1}/15)")
            await asyncio.sleep(2)
            
    if not connected:
        print("❌ CRITICAL: Could not connect to MCP Server after multiple attempts. Exiting.")
        sys.exit(1)

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