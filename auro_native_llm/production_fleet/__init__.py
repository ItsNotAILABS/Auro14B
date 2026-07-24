"""NOVA-governed production inference and internal-agent runtime.

Exports are loaded lazily so an optional wallet, SDK, or document dependency
cannot prevent unrelated browser, office, or model-runtime modules from loading.
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

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
from .runtime import AgentManager, ModelEndpoint, NovaRuntime
from .organ_sdk import AuroOrganSDK, SDKConfig
from .capabilities import NativeCapabilities

__all__ = ["AgentManager", "ModelEndpoint", "NovaRuntime", "AuroOrganSDK", "SDKConfig", "NativeCapabilities"]
from .receipts import ReceiptLedger
from .wallet import PaperWallet
from .office import NativeOffice
from .vault import IntegrityVault

__all__ = ["AgentManager", "ModelEndpoint", "NovaRuntime", "AuroOrganSDK", "SDKConfig", "NativeCapabilities", "ReceiptLedger"]
from .wallet import PaperWallet
from .office import NativeOffice
from .vault import IntegrityVault

__all__ = ["AgentManager", "ModelEndpoint", "NovaRuntime", "AuroOrganSDK", "SDKConfig", "NativeCapabilities", "ReceiptLedger", "PaperWallet", "NativeOffice", "IntegrityVault"]
