"""Chrome DevTools Protocol client — navigate, DOM, input, evaluate.

Works against Chrome launched with --remote-debugging-port.
Async when ``websockets`` is installed; HTTP list/new-tab always available.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ChromeUnavailable(RuntimeError):
    """Chrome CDP endpoint not reachable."""


@dataclass
class DOMSnapshot:
    url: str
    title: str
    node_count: int
    interactive: List[Dict[str, Any]]
    text_preview: str
    raw_nodes: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "node_count": self.node_count,
            "interactive": self.interactive[:80],
            "text_preview": self.text_preview[:4000],
        }

    def compress_for_llm(self, max_chars: int = 3500) -> str:
        lines = [
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Nodes: {self.node_count}",
            "Interactive:",
        ]
        for el in self.interactive[:40]:
            lines.append(
                f"  - [{el.get('tag')}] id={el.get('id','')} name={el.get('name','')} "
                f"text={el.get('text','')[:60]!r} href={el.get('href','')[:80]}"
            )
        lines.append("Text:")
        lines.append(self.text_preview[: max_chars // 2])
        out = "\n".join(lines)
        return out[:max_chars]


def _http_json(url: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_put(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, method="PUT")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ChromeTab:
    """One Chrome target via CDP WebSocket."""

    def __init__(self, target_id: str, ws_url: str) -> None:
        self.target_id = target_id
        self.ws_url = ws_url
        self._ws = None
        self._msg_id = 1
        self._pending: Dict[int, Any] = {}
        self._loop_task = None
        self.mock = False
        self._mock_url = "about:blank"
        self._mock_dom: List[Dict[str, Any]] = []

    @classmethod
    def mock_tab(cls, target_id: str = "mock-tab") -> "ChromeTab":
        t = cls(target_id, "ws://mock")
        t.mock = True
        t._mock_dom = [
            {"nodeName": "A", "attributes": ["href", "https://example.com"], "nodeValue": ""},
            {"nodeName": "BUTTON", "attributes": ["id", "go"], "nodeValue": "Go"},
            {"nodeName": "INPUT", "attributes": ["name", "q", "type", "text"], "nodeValue": ""},
            {"nodeName": "#text", "attributes": [], "nodeValue": "Example Domain mock page"},
        ]
        return t

    async def connect(self) -> None:
        if self.mock:
            return
        import asyncio
        import websockets

        self._ws = await websockets.connect(self.ws_url, ping_interval=None, max_size=10**8)
        self._loop_task = asyncio.create_task(self._listen())
        for domain in ("Page", "DOM", "Runtime", "Network", "Input"):
            await self.send(f"{domain}.enable")

    async def _listen(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                data = json.loads(message)
                mid = data.get("id")
                if mid in self._pending:
                    fut = self._pending.pop(mid)
                    if not fut.done():
                        fut.set_result(data)
        except Exception:
            pass

    async def send(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.mock:
            return self._mock_send(method, params or {})
        import asyncio

        if not self._ws:
            raise ChromeUnavailable("tab not connected")
        mid = self._msg_id
        self._msg_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        try:
            data = await asyncio.wait_for(fut, timeout=20.0)
        except asyncio.TimeoutError:
            self._pending.pop(mid, None)
            raise TimeoutError(f"CDP timeout: {method}")
        if "error" in data:
            raise RuntimeError(data["error"])
        return data.get("result", {})

    def _mock_send(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "Page.navigate":
            self._mock_url = params.get("url", self._mock_url)
            return {"frameId": "mock"}
        if method == "Runtime.evaluate":
            expr = params.get("expression", "")
            if "document.title" in expr:
                return {"result": {"value": "Mock Page"}}
            if "location.href" in expr:
                return {"result": {"value": self._mock_url}}
            if "innerText" in expr:
                return {"result": {"value": "Example Domain mock page content for agents."}}
            return {"result": {"value": None}}
        if method == "DOM.getFlattenedDocument":
            return {"nodes": self._mock_dom}
        if method.startswith("Input."):
            return {}
        if method.endswith(".enable"):
            return {}
        return {}

    async def navigate(self, url: str) -> Dict[str, Any]:
        return await self.send("Page.navigate", {"url": url})

    async def evaluate(self, expression: str) -> Any:
        res = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        return res.get("result", {}).get("value")

    async def click_xy(self, x: float, y: float) -> None:
        await self.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        await self.send(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1},
        )
        await self.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
        )

    async def type_text(self, text: str) -> None:
        for ch in text:
            await self.send("Input.dispatchKeyEvent", {"type": "char", "text": ch})

    async def snapshot(self) -> DOMSnapshot:
        url = str(await self.evaluate("location.href") or "")
        title = str(await self.evaluate("document.title") or "")
        text = str(
            await self.evaluate(
                "(document.body && document.body.innerText) ? document.body.innerText.slice(0, 8000) : ''"
            )
            or ""
        )
        nodes_res = await self.send("DOM.getFlattenedDocument", {"depth": -1, "pierce": True})
        nodes = nodes_res.get("nodes") or []
        interactive: List[Dict[str, Any]] = []
        for n in nodes:
            tag = str(n.get("nodeName", "")).upper()
            if tag in ("A", "BUTTON", "INPUT", "TEXTAREA", "SELECT", "SUMMARY"):
                attrs = n.get("attributes") or []
                ad = {attrs[i]: attrs[i + 1] for i in range(0, len(attrs) - 1, 2)}
                interactive.append(
                    {
                        "tag": tag.lower(),
                        "id": ad.get("id", ""),
                        "name": ad.get("name", ""),
                        "type": ad.get("type", ""),
                        "href": ad.get("href", ""),
                        "text": (n.get("nodeValue") or ad.get("value") or ad.get("placeholder") or "")[:80],
                    }
                )
        return DOMSnapshot(
            url=url,
            title=title,
            node_count=len(nodes),
            interactive=interactive,
            text_preview=text,
            raw_nodes=nodes[:200],
        )

    async def close(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        if self._ws:
            await self._ws.close()


class ChromeCDP:
    """Browser controller: spawn/connect Chrome, open tabs, DOM protocol."""

    def __init__(
        self,
        port: int = 9222,
        headless: bool = True,
        user_data_dir: Optional[str] = None,
        mock: bool = False,
    ) -> None:
        self.port = port
        self.headless = headless
        self.user_data_dir = user_data_dir or os.path.join(
            os.path.expanduser("~"), ".auro_chrome_data"
        )
        self.mock = mock
        self.process: Optional[subprocess.Popen] = None
        self.tabs: Dict[str, ChromeTab] = {}

    def _chrome_path(self) -> str:
        """Resolve Chrome or Edge on Windows/macOS/Linux (incl. ARM64 paths)."""
        system = platform.system()
        env = os.environ.get("AURO_CHROME_PATH") or os.environ.get("CHROME_PATH")
        if env and os.path.exists(env):
            return env
        if system == "Windows":
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                # Edge often present when Chrome is not
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
                # Chromium / Brave
                os.path.expandvars(r"%LOCALAPPDATA%\Chromium\Application\chrome.exe"),
                os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    return p
            # last resort: PATH
            import shutil

            for name in ("chrome", "msedge", "brave", "chromium"):
                found = shutil.which(name)
                if found:
                    return found
            return "chrome"
        if system == "Darwin":
            for p in (
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
            ):
                if os.path.exists(p):
                    return p
            return "google-chrome"
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge"):
            import shutil

            found = shutil.which(name)
            if found:
                return found
        return "google-chrome"

    def is_available(self) -> bool:
        if self.mock:
            return True
        try:
            _http_json(f"http://127.0.0.1:{self.port}/json/version", timeout=1.5)
            return True
        except Exception:
            return False

    def start_chrome(self) -> None:
        if self.mock:
            return
        if self.is_available():
            return
        os.makedirs(self.user_data_dir, exist_ok=True)
        args = [
            self._chrome_path(),
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-background-networking",
            "--mute-audio",
        ]
        if self.headless:
            args.append("--headless=new")
        self.process = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(20):
            if self.is_available():
                return
            time.sleep(0.25)
        raise ChromeUnavailable(f"Chrome did not open CDP on port {self.port}")

    def list_targets(self) -> List[Dict[str, Any]]:
        if self.mock:
            return [{"id": "mock-tab", "type": "page", "webSocketDebuggerUrl": "ws://mock"}]
        return _http_json(f"http://127.0.0.1:{self.port}/json")

    async def connect_tab(self, target_id: Optional[str] = None) -> ChromeTab:
        if self.mock:
            tab = ChromeTab.mock_tab(target_id or "mock-tab")
            await tab.connect()
            self.tabs[tab.target_id] = tab
            return tab
        targets = self.list_targets()
        pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
        if target_id:
            pages = [t for t in pages if t.get("id") == target_id]
        if not pages:
            # open new tab
            try:
                t = _http_json(f"http://127.0.0.1:{self.port}/json/new?about:blank")
            except Exception:
                t = _http_put(f"http://127.0.0.1:{self.port}/json/new?about:blank")
            pages = [t]
        t0 = pages[0]
        tab = ChromeTab(str(t0["id"]), str(t0["webSocketDebuggerUrl"]))
        await tab.connect()
        self.tabs[tab.target_id] = tab
        return tab

    async def close(self) -> None:
        for tab in list(self.tabs.values()):
            await tab.close()
        self.tabs.clear()
        if self.process:
            self.process.terminate()
            self.process = None
