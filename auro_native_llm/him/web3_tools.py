"""HIM standard web3 tooling — Node/npm applet packages + secure /api/* client.

Secrets never leave him-web3/.env on the Node server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_WEB3 = _REPO / "him-web3"
_DEFAULT_BASE = os.environ.get("HIM_WEB3_URL", "http://127.0.0.1:8787")


class HimWeb3Tools:
    """Agentic tools for HIM: install packages, call secure API routes."""

    def __init__(
        self,
        root: Optional[Path] = None,
        api_base: str = _DEFAULT_BASE,
    ) -> None:
        self.root = Path(root or _WEB3)
        self.api_base = api_base.rstrip("/")

    def status(self) -> Dict[str, Any]:
        return {
            "ok": self.root.exists(),
            "name": "HIM",
            "web3_root": str(self.root),
            "node": shutil.which("node"),
            "npm": shutil.which("npm") or shutil.which("npm.cmd"),
            "api_base": self.api_base,
            "package_json": (self.root / "package.json").exists(),
            "env_example": (self.root / ".env.example").exists(),
            "env_present": (self.root / ".env").exists(),
            "security": "RPC keys only in him-web3/.env — React uses /api/* only",
            "install": "node him-web3/scripts/install_applet_package.js <pkg>",
        }

    def install_applet_package(self, packages: List[str]) -> Dict[str, Any]:
        """Install ethers/viem/web3.js etc. into him-web3 via npm."""
        if not packages:
            return {"ok": False, "error": "packages list empty"}
        if not self.root.exists():
            return {"ok": False, "error": f"missing {self.root}"}
        script = self.root / "scripts" / "install_applet_package.js"
        node = shutil.which("node")
        if not node:
            return {"ok": False, "error": "node not on PATH", "status": "TECHNICALLY_UNAVAILABLE"}
        try:
            p = subprocess.run(
                [node, str(script), *packages],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=300,
            )
            return {
                "ok": p.returncode == 0,
                "packages": packages,
                "returncode": p.returncode,
                "stdout": (p.stdout or "")[-2000:],
                "stderr": (p.stderr or "")[-1000:],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:300]}

    def npm_setup(self) -> Dict[str, Any]:
        """npm install at him-web3 root."""
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            return {"ok": False, "error": "npm not on PATH"}
        try:
            p = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit"],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {
                "ok": p.returncode == 0,
                "returncode": p.returncode,
                "stdout": (p.stdout or "")[-1500:],
                "stderr": (p.stderr or "")[-800:],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:300]}

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                body = r.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                return json.loads(body)
            except Exception:
                return {"ok": False, "error": f"HTTP {e.code}", "url": url}
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc)[:200],
                "url": url,
                "hint": "Start server: cd him-web3 && npm run server",
            }

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.api_base}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200], "url": url}

    def api_health(self) -> Dict[str, Any]:
        return self._get("/api/health")

    def block_number(self) -> Dict[str, Any]:
        return self._get("/api/chain/block-number")

    def balance(self, address: str) -> Dict[str, Any]:
        return self._get(f"/api/chain/balance/{address}")

    def latest_block(self) -> Dict[str, Any]:
        return self._get("/api/chain/block?tag=latest")

    def contract_call(
        self,
        address: str,
        abi: list,
        function_name: str,
        args: Optional[list] = None,
    ) -> Dict[str, Any]:
        return self._post(
            "/api/chain/call",
            {
                "address": address,
                "abi": abi,
                "functionName": function_name,
                "args": args or [],
            },
        )
