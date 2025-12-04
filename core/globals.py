from typing import List, Any
from langchain_core.tools import BaseTool

# This list will be populated at startup by main.py
mcp_tools: List[BaseTool] = []

graph_app: Any = None