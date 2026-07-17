"""Pythonista-inspired host for Auro.

Model (from the Pythonista app + succotash windows-runtime-sdk):

  JS / browser services  = foreground shell (render UI, DB tables, events)
  Python scripts         = background intelligence (train, agents, engines)
  ui.*                   = Python declares views → JSON UI tree → JS renders
  db.*                   = Python writes rows → table schemas → JS table UI

Not iOS-locked — same pattern as Pythonista's ``ui`` module + background
scripts, adapted for MESIE/Auro + potential-succotash engines.
"""

from auro_native_llm.pythonista.ui import (
    View,
    Button,
    Label,
    TextField,
    TextView,
    TableView,
    StackView,
    WebView,
    ui_tree,
    reset_ui,
)
from auro_native_llm.pythonista.runtime import ScriptRuntime, run_script
from auro_native_llm.pythonista.service import PythonistaService, get_service
from auro_native_llm.pythonista.db import PythonistaDB

__all__ = [
    "Button",
    "Label",
    "PythonistaDB",
    "PythonistaService",
    "ScriptRuntime",
    "StackView",
    "TableView",
    "TextField",
    "TextView",
    "View",
    "WebView",
    "get_service",
    "reset_ui",
    "run_script",
    "ui_tree",
]
