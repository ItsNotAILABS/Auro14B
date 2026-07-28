from __future__ import annotations

import ast
from pathlib import Path


def test_production_fleet_imports_without_initializing_optional_organs():
    import auro_native_llm.production_fleet as fleet
    from auro_native_llm.production_fleet.capabilities import BUILTINS, NativeCapabilities
    from auro_native_llm.production_fleet.server import Handler

    assert fleet.NativeCapabilities is NativeCapabilities
    assert Handler.runtime is None
    names = [item.name for item in BUILTINS]
    assert len(names) == len(set(names))
    assert {"browser.task.enqueue", "wallet.transfer_paper", "office.create_bundle"} <= set(names)


def test_capability_and_server_sources_have_single_canonical_classes_and_handlers():
    capability_tree = ast.parse(Path("auro_native_llm/production_fleet/capabilities.py").read_text())
    server_tree = ast.parse(Path("auro_native_llm/production_fleet/server.py").read_text())

    capability_classes = [node for node in capability_tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeCapabilities"]
    handler_classes = [node for node in server_tree.body if isinstance(node, ast.ClassDef) and node.name == "Handler"]
    assert len(capability_classes) == 1
    assert len(handler_classes) == 1

    methods = [node.name for node in handler_classes[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert methods.count("do_GET") == 1
    assert methods.count("do_POST") == 1
    assert methods.count("_json") == 1
    assert methods.count("_bytes") == 1


def test_cloudflare_chat_contract_requires_identity_auth_and_quota():
    source = Path("cloudflare-platform/src/public-api.ts").read_text()
    assert "authenticated chat token required" in source
    assert "x-session-id must be a stable" in source
    assert "await stub.consume" in source
    assert "hosted compatibility lane; not an AURO-trained checkpoint" in source
