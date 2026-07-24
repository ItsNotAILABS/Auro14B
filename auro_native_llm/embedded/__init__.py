"""Embedded capability organs — Monaco, Jupyter, online search, self-hosted MCP.

These are not product features bolted on: they are tools every AuroMind is born with.
"""

from auro_native_llm.embedded.monaco import MonacoSession, MonacoOrgan
from auro_native_llm.embedded.jupyter import NotebookSession, JupyterOrgan
from auro_native_llm.embedded.search import OnlineSearch, SearchOrgan
from auro_native_llm.embedded.mcp_hub import MCPHub, MCPOrgan
from auro_native_llm.embedded.teach import ToolCurriculum

__all__ = [
    "JupyterOrgan",
    "MCPHub",
    "MCPOrgan",
    "MonacoOrgan",
    "MonacoSession",
    "NotebookSession",
    "OnlineSearch",
    "SearchOrgan",
    "ToolCurriculum",
]
