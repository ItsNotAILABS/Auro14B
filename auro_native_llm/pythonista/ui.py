"""Pythonista-like ``ui`` module — declarative views for JS host rendering.

Python scripts build UIs with familiar names (View, Button, Label, TableView).
Nothing draws on-device: each control serializes to a JSON node the JS service
renders as HTML/DOM (same role as Pythonista's native UI bridge).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

_id_counter = itertools.count(1)
_ROOT: Optional["View"] = None
_EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {}


def _nid(prefix: str = "n") -> str:
    return f"{prefix}_{next(_id_counter)}"


def reset_ui() -> None:
    global _ROOT
    _ROOT = None
    _EVENT_HANDLERS.clear()


def ui_tree() -> Dict[str, Any]:
    if _ROOT is None:
        return {"type": "empty", "children": [], "schema": "auro.pythonista.ui.v1"}
    return {
        "schema": "auro.pythonista.ui.v1",
        "root": _ROOT.to_dict(),
        "ts": time.time(),
    }


def register_handler(node_id: str, action: str, fn: Callable[..., Any]) -> None:
    _EVENT_HANDLERS[f"{node_id}:{action}"] = fn


def dispatch_event(node_id: str, action: str, payload: Optional[Dict[str, Any]] = None) -> Any:
    key = f"{node_id}:{action}"
    fn = _EVENT_HANDLERS.get(key)
    if fn is None:
        return {"ok": False, "error": f"no handler for {key}"}
    try:
        return {"ok": True, "result": fn(payload or {})}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@dataclass
class View:
    """Base view — Pythonista ``ui.View`` analogue."""

    name: str = ""
    background_color: str = "#141b2d"
    flex: str = "WH"
    frame: Optional[List[float]] = None  # x,y,w,h
    children: List["View"] = field(default_factory=list)
    props: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: _nid("view"))
    kind: str = "View"

    def add_subview(self, child: "View") -> "View":
        self.children.append(child)
        return self

    def present(self, style: str = "sheet") -> "View":
        """Set this view as the active root for the JS host."""
        global _ROOT
        _ROOT = self
        self.props["present_style"] = style
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.kind,
            "name": self.name,
            "background_color": self.background_color,
            "flex": self.flex,
            "frame": self.frame,
            "props": dict(self.props),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Label(View):
    text: str = ""
    text_color: str = "#e8eefc"
    font_size: int = 14
    alignment: str = "left"  # left|center|right
    kind: str = "Label"

    def __post_init__(self) -> None:
        self.props.update(
            {
                "text": self.text,
                "text_color": self.text_color,
                "font_size": self.font_size,
                "alignment": self.alignment,
            }
        )


@dataclass
class Button(View):
    title: str = "Button"
    action: Optional[Callable[..., Any]] = None
    tint_color: str = "#5b8cff"
    kind: str = "Button"

    def __post_init__(self) -> None:
        self.props.update({"title": self.title, "tint_color": self.tint_color})
        if self.action is not None:
            register_handler(self.node_id, "action", self.action)

    def set_action(self, fn: Callable[..., Any]) -> "Button":
        self.action = fn
        register_handler(self.node_id, "action", fn)
        return self


@dataclass
class TextField(View):
    placeholder: str = ""
    text: str = ""
    kind: str = "TextField"

    def __post_init__(self) -> None:
        self.props.update({"placeholder": self.placeholder, "text": self.text})


@dataclass
class TextView(View):
    text: str = ""
    editable: bool = True
    kind: str = "TextView"

    def __post_init__(self) -> None:
        self.props.update({"text": self.text, "editable": self.editable})


@dataclass
class TableView(View):
    """Renders as a data table in the JS host (Pythonista table analogue)."""

    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    source_table: str = ""  # optional DB table name
    kind: str = "TableView"

    def __post_init__(self) -> None:
        self.props.update(
            {
                "columns": list(self.columns),
                "rows": list(self.rows),
                "source_table": self.source_table,
            }
        )

    def reload(self, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
        self.rows = [dict(r) for r in rows]
        if columns is not None:
            self.columns = list(columns)
        self.props["rows"] = list(self.rows)
        self.props["columns"] = list(self.columns)


@dataclass
class StackView(View):
    axis: str = "vertical"  # vertical|horizontal
    spacing: int = 8
    kind: str = "StackView"

    def __post_init__(self) -> None:
        self.props.update({"axis": self.axis, "spacing": self.spacing})


@dataclass
class WebView(View):
    html: str = ""
    url: str = ""
    kind: str = "WebView"

    def __post_init__(self) -> None:
        self.props.update({"html": self.html, "url": self.url})


# Module-level convenience aliases matching Pythonista style
def make_button(title: str, action: Optional[Callable[..., Any]] = None) -> Button:
    return Button(title=title, action=action)


def make_label(text: str) -> Label:
    return Label(text=text)
