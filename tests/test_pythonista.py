"""Pythonista-style host: Python UI → JS tree, DB tables, background jobs."""

from __future__ import annotations

from auro_native_llm.pythonista.runtime import run_script
from auro_native_llm.pythonista.service import PythonistaService, get_service
from auro_native_llm.pythonista.ui import Button, Label, StackView, TableView, View, ui_tree, reset_ui


def test_ui_tree_serialization():
    reset_ui()
    root = View(name="r")
    stack = StackView(axis="vertical")
    stack.add_subview(Label(text="hello"))
    stack.add_subview(Button(title="Go"))
    stack.add_subview(TableView(columns=["a"], rows=[{"a": 1}]))
    root.add_subview(stack)
    root.present()
    tree = ui_tree()
    assert tree["schema"] == "auro.pythonista.ui.v1"
    assert tree["root"]["type"] == "View"
    kids = tree["root"]["children"][0]["children"]
    types = [k["type"] for k in kids]
    assert "Label" in types and "Button" in types and "TableView" in types


def test_run_hello_dashboard_script():
    svc = PythonistaService()
    res = svc.run_script(script_name="hello_dashboard.py")
    assert res["ok"] is True, res.get("error")
    assert res["ui"]["root"]["type"] == "View"
    assert any(t.get("table") == "notes" for t in res.get("tables") or [])
    # event on button
    btn = None

    def find_button(node):
        nonlocal btn
        if node.get("type") == "Button":
            btn = node
            return
        for c in node.get("children") or []:
            find_button(c)

    find_button(res["ui"]["root"])
    assert btn is not None
    ev = svc.event(btn["id"], "action", {"text": "from-test"})
    assert ev.get("ok") is True
    notes = svc.table("notes")
    assert notes["ok"] is True
    assert notes["count"] >= 2


def test_background_job():
    svc = get_service()
    out = svc.run_script(script_name="bg_pulse.py", background=True)
    assert out["ok"] is True
    assert out.get("background") is True
    assert out["job"]["job_id"]


def test_inline_script_with_import_ui():
    src = """
import ui
v = ui.View(name='x')
v.add_subview(ui.Label(text='inline'))
v.present()
print('ok')
"""
    res = run_script(src)
    assert res["ok"] is True, res.get("error")
    assert res["stdout"].strip() == "ok"
    assert res["ui"]["root"]["children"][0]["props"]["text"] == "inline"
