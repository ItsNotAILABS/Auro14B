"""Lightweight DB for Pythonista scripts — rows render as JS table UIs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class PythonistaDB:
    """SQLite store with render-friendly table schemas."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path or (Path.home() / ".auro_pythonista" / "app.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_meta()

    def _ensure_meta(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _py_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        self._conn.commit()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Dict[str, Any]:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return {"ok": True, "rowcount": cur.rowcount, "lastrowid": cur.lastrowid}

    def create_table(self, name: str, columns: Dict[str, str]) -> Dict[str, Any]:
        """columns: {col_name: SQL_TYPE}"""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        parts = ", ".join(f"{k} {v}" for k, v in columns.items())
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {safe} ({parts})")
        self._conn.commit()
        # remember render schema
        self._conn.execute(
            "INSERT OR REPLACE INTO _py_meta(key, value) VALUES (?, ?)",
            (f"schema:{safe}", json.dumps({"columns": list(columns.keys()), "types": columns})),
        )
        self._conn.commit()
        return {"ok": True, "table": safe, "columns": list(columns.keys())}

    def insert(self, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_sql = ", ".join(cols)
        self._conn.execute(
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
            [row[c] for c in cols],
        )
        self._conn.commit()
        return {"ok": True, "table": table}

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def table_render(self, table: str, limit: int = 200) -> Dict[str, Any]:
        """Schema + rows shaped for JS TableView / data-grid."""
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in table)
        try:
            rows = self.query(f"SELECT * FROM {safe} LIMIT ?", (limit,))
        except Exception as exc:
            return {"ok": False, "error": str(exc), "table": table}
        columns = list(rows[0].keys()) if rows else self._columns_for(safe)
        return {
            "ok": True,
            "schema": "auro.pythonista.table.v1",
            "table": safe,
            "columns": columns,
            "rows": rows,
            "count": len(rows),
            "ts": time.time(),
        }

    def _columns_for(self, table: str) -> List[str]:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    def list_tables(self) -> List[str]:
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '_py_%'"
        )
        return [r["name"] for r in rows]

    def close(self) -> None:
        self._conn.close()
