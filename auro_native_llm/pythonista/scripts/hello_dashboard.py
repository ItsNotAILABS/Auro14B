"""Example Pythonista-style script: declare UI + seed a DB table.

Run via JS host / API:
  POST /v1/pythonista/run {"script_name": "hello_dashboard.py"}
"""

import ui
import db
from datetime import datetime

# --- database (renders as JS table) ---
db.create_table(
    "notes",
    {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "title": "TEXT",
        "body": "TEXT",
        "created": "TEXT",
    },
)
db.insert(
    "notes",
    {
        "title": "MESIE native",
        "body": "Python script runs in background under JS service",
        "created": datetime.utcnow().isoformat() + "Z",
    },
)
db.insert(
    "notes",
    {
        "title": "Pythonista pattern",
        "body": "ui.View → JSON → JS renders DOM",
        "created": datetime.utcnow().isoformat() + "Z",
    },
)

rows = db.query("SELECT id, title, body, created FROM notes ORDER BY id DESC LIMIT 20")

# --- UI declared in Python (host renders) ---
root = ui.View(name="dashboard", background_color="#0b1020")
stack = ui.StackView(axis="vertical", spacing=10)
stack.add_subview(ui.Label(text="Auro · Pythonista host", font_size=18, text_color="#5b8cff"))
stack.add_subview(
    ui.Label(
        text="Python builds this UI; JS service paints it. DB table below.",
        font_size=13,
        text_color="#8b9bb8",
    )
)

field = ui.TextField(placeholder="New note title…", name="title_field")
stack.add_subview(field)


def on_add(payload):
    title = (payload or {}).get("text") or field.props.get("text") or "untitled"
    db.insert(
        "notes",
        {
            "title": title,
            "body": f"added from UI event @ {datetime.utcnow().isoformat()}Z",
            "created": datetime.utcnow().isoformat() + "Z",
        },
    )
    console.hud_alert(f"saved: {title}")
    return {"saved": title}


btn = ui.Button(title="Add note", action=on_add, tint_color="#3dd68c")
stack.add_subview(btn)

table = ui.TableView(
    name="notes_table",
    columns=["id", "title", "body", "created"],
    rows=rows,
    source_table="notes",
)
stack.add_subview(table)
root.add_subview(stack)
root.present("sheet")

print("hello_dashboard ready — UI presented, notes table seeded")
