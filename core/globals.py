from typing import List
from langchain_core.tools import BaseTool

# This list will be populated at startup by main.py
mcp_tools: List[BaseTool] = []