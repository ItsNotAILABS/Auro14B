"""Embedded Jupyter notebook organ — cells, execute, .ipynb export.

Uses safe Python exec for code cells (same restricted sandbox as work agent).
Full Jupyter kernel optional if jupyter is installed.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from auro_native_llm.work.safe_exec import safe_exec_python


@dataclass
class Cell:
    cell_id: str
    cell_type: str  # code | markdown
    source: str
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_count: Optional[int] = None

    def to_ipynb(self) -> Dict[str, Any]:
        if self.cell_type == "markdown":
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": self.source.splitlines(keepends=True) or [""],
            }
        outs = []
        for o in self.outputs:
            if o.get("ok"):
                text = json.dumps(o.get("locals", o), indent=2)
                outs.append(
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": text.splitlines(keepends=True) or [""],
                    }
                )
            else:
                outs.append(
                    {
                        "output_type": "error",
                        "ename": "Error",
                        "evalue": str(o.get("error", "")),
                        "traceback": [str(o.get("error", ""))],
                    }
                )
        return {
            "cell_type": "code",
            "execution_count": self.execution_count,
            "metadata": {},
            "source": self.source.splitlines(keepends=True) or [""],
            "outputs": outs,
        }


@dataclass
class NotebookSession:
    notebook_id: str
    title: str = "Auro Notebook"
    cells: List[Cell] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    exec_counter: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notebook_id": self.notebook_id,
            "title": self.title,
            "n_cells": len(self.cells),
            "cells": [
                {
                    "cell_id": c.cell_id,
                    "cell_type": c.cell_type,
                    "source": c.source,
                    "outputs": c.outputs,
                    "execution_count": c.execution_count,
                }
                for c in self.cells
            ],
        }

    def to_ipynb(self) -> Dict[str, Any]:
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Auro Safe Python",
                    "language": "python",
                    "name": "auro_python",
                },
                "language_info": {"name": "python"},
                "auro": {"embedded": True, "compute_plane": "MESIE"},
            },
            "cells": [c.to_ipynb() for c in self.cells],
        }


class JupyterOrgan:
    """Embedded notebook organ for every Auro mind."""

    def __init__(self) -> None:
        self.notebooks: Dict[str, NotebookSession] = {}

    def create(self, title: str = "Auro Notebook", notebook_id: Optional[str] = None) -> Dict[str, Any]:
        nid = notebook_id or f"nb-{uuid.uuid4().hex[:10]}"
        nb = NotebookSession(notebook_id=nid, title=title)
        # starter cells
        nb.cells.append(
            Cell(
                cell_id=f"c-{uuid.uuid4().hex[:8]}",
                cell_type="markdown",
                source=f"# {title}\n\nEmbedded Auro notebook — MESIE compute plane.",
            )
        )
        nb.cells.append(
            Cell(
                cell_id=f"c-{uuid.uuid4().hex[:8]}",
                cell_type="code",
                source="x = 1 + 1\nx",
            )
        )
        self.notebooks[nid] = nb
        return {"ok": True, "action": "jupyter.create", "notebook": nb.to_dict()}

    def get(self, notebook_id: str) -> Dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        if not nb:
            return {"ok": False, "error": f"unknown notebook {notebook_id}"}
        return {"ok": True, "action": "jupyter.get", "notebook": nb.to_dict()}

    def add_cell(
        self,
        notebook_id: str,
        source: str,
        *,
        cell_type: str = "code",
        index: Optional[int] = None,
    ) -> Dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        if not nb:
            return {"ok": False, "error": f"unknown notebook {notebook_id}"}
        cell = Cell(
            cell_id=f"c-{uuid.uuid4().hex[:8]}",
            cell_type=cell_type if cell_type in ("code", "markdown") else "code",
            source=source,
        )
        if index is None or index >= len(nb.cells):
            nb.cells.append(cell)
        else:
            nb.cells.insert(max(0, index), cell)
        return {"ok": True, "action": "jupyter.add_cell", "cell_id": cell.cell_id, "notebook": nb.to_dict()}

    def execute_cell(self, notebook_id: str, cell_id: str) -> Dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        if not nb:
            return {"ok": False, "error": f"unknown notebook {notebook_id}"}
        cell = next((c for c in nb.cells if c.cell_id == cell_id), None)
        if not cell:
            return {"ok": False, "error": f"unknown cell {cell_id}"}
        if cell.cell_type != "code":
            return {"ok": True, "action": "jupyter.execute", "skipped": "markdown"}
        # strip trailing bare expression display: assign to _ if single expr line
        code = cell.source
        result = safe_exec_python(code)
        nb.exec_counter += 1
        cell.execution_count = nb.exec_counter
        cell.outputs = [result]
        return {
            "ok": bool(result.get("ok")),
            "action": "jupyter.execute",
            "cell_id": cell_id,
            "result": result,
            "execution_count": cell.execution_count,
        }

    def execute_all(self, notebook_id: str) -> Dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        if not nb:
            return {"ok": False, "error": f"unknown notebook {notebook_id}"}
        results = []
        for c in nb.cells:
            if c.cell_type == "code":
                results.append(self.execute_cell(notebook_id, c.cell_id))
        return {"ok": all(r.get("ok") for r in results) if results else True, "results": results}

    def export_ipynb(self, notebook_id: str) -> Dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        if not nb:
            return {"ok": False, "error": f"unknown notebook {notebook_id}"}
        return {"ok": True, "action": "jupyter.export", "ipynb": nb.to_ipynb()}

    def list_notebooks(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "notebooks": [
                {"notebook_id": n.notebook_id, "title": n.title, "n_cells": len(n.cells)}
                for n in self.notebooks.values()
            ],
        }
