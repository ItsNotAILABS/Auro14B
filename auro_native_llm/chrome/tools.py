"""Chrome toolbelt — sync facade for work agents (runs async CDP under the hood)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from auro_native_llm.chrome.cdp import ChromeCDP, ChromeTab, ChromeUnavailable, DOMSnapshot


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # nested: create new loop in thread would be heavier; use asyncio.run when possible
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class ChromeToolbelt:
    """High-level Chrome actions for Auro work agents."""

    def __init__(
        self,
        port: int = 9222,
        headless: bool = True,
        mock: bool = False,
        auto_start: bool = False,
    ) -> None:
        self.cdp = ChromeCDP(port=port, headless=headless, mock=mock)
        self.tab: Optional[ChromeTab] = None
        self.auto_start = auto_start
        self.last_dom: Optional[DOMSnapshot] = None

    def ensure(self) -> None:
        if self.cdp.mock or self.cdp.is_available():
            return
        if self.auto_start:
            self.cdp.start_chrome()
        elif not self.cdp.is_available():
            raise ChromeUnavailable(
                "Chrome CDP not available. Start Chrome with "
                f"--remote-debugging-port={self.cdp.port} or pass mock=True / auto_start=True"
            )

    def _ensure_tab(self) -> ChromeTab:
        self.ensure()
        if self.tab is None:
            self.tab = _run(self.cdp.connect_tab())
        return self.tab

    def navigate(self, url: str) -> Dict[str, Any]:
        tab = self._ensure_tab()
        _run(tab.navigate(url))
        return {"ok": True, "action": "navigate", "url": url}

    def dom(self) -> Dict[str, Any]:
        tab = self._ensure_tab()
        snap = _run(tab.snapshot())
        self.last_dom = snap
        return {"ok": True, "action": "dom", "snapshot": snap.to_dict(), "llm": snap.compress_for_llm()}

    def click(self, x: float, y: float) -> Dict[str, Any]:
        tab = self._ensure_tab()
        _run(tab.click_xy(x, y))
        return {"ok": True, "action": "click", "x": x, "y": y}

    def type_text(self, text: str) -> Dict[str, Any]:
        tab = self._ensure_tab()
        _run(tab.type_text(text))
        return {"ok": True, "action": "type", "text": text[:200]}

    def evaluate(self, js: str) -> Dict[str, Any]:
        tab = self._ensure_tab()
        value = _run(tab.evaluate(js))
        return {"ok": True, "action": "evaluate", "value": value}

    def health(self) -> Dict[str, Any]:
        return {
            "chrome_available": self.cdp.is_available() or self.cdp.mock,
            "mock": self.cdp.mock,
            "port": self.cdp.port,
            "tab": self.tab.target_id if self.tab else None,
        }

    def close(self) -> None:
        if self.tab or self.cdp.tabs:
            _run(self.cdp.close())
        self.tab = None
