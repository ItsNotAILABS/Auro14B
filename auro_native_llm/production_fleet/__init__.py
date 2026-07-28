"""NOVA-governed production inference and internal-agent runtime.

Exports are loaded lazily so optional organs cannot prevent unrelated runtime
modules from importing.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AgentManager": (".runtime", "AgentManager"),
    "ModelEndpoint": (".runtime", "ModelEndpoint"),
    "NovaRuntime": (".runtime", "NovaRuntime"),
    "AuroOrganSDK": (".organ_sdk", "AuroOrganSDK"),
    "SDKConfig": (".organ_sdk", "SDKConfig"),
    "NativeCapabilities": (".capabilities", "NativeCapabilities"),
    "ReceiptLedger": (".receipts", "ReceiptLedger"),
    "PaperWallet": (".wallet", "PaperWallet"),
    "NativeOffice": (".office", "NativeOffice"),
    "IntegrityVault": (".vault", "IntegrityVault"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
