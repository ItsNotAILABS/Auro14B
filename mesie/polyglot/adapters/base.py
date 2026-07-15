"""Base adapter for AIS Vector Polyglot runtimes."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

from mesie.polyglot.contract import (
    AISVectorMessage,
    AISVectorResponse,
    PolyglotAction,
    RuntimeId,
)


class PolyglotAdapter(ABC):
    runtime: RuntimeId

    @abstractmethod
    def available(self) -> bool:
        """True when native runtime is installed; False uses fallback."""

    @property
    def mode(self) -> str:
        return "native" if self.available() else "fallback"

    def dispatch(self, message: AISVectorMessage) -> AISVectorResponse:
        t0 = time.perf_counter()
        try:
            data, vector = self._handle(message)
            ms = (time.perf_counter() - t0) * 1000
            return AISVectorResponse(
                ok=True,
                runtime=self.runtime,
                action=message.action,
                data=data,
                vector=vector,
                latency_ms=ms,
            )
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            return AISVectorResponse(
                ok=False,
                runtime=self.runtime,
                action=message.action,
                latency_ms=ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    @abstractmethod
    def _handle(self, message: AISVectorMessage) -> tuple[dict, Optional[list[float]]]:
        ...

    def health(self) -> dict:
        return {"runtime": self.runtime.value, "mode": self.mode, "available": self.available()}