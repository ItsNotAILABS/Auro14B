from __future__ import annotations

import json
from pathlib import Path

import pytest

import auro_native_llm.sovereign.contract as contract_module
from auro_native_llm.sovereign import bind_sovereign


def _fixture(root: Path) -> Path:
    required = [
        "AGENTS.md",
        "LICENSE",
        "cpl/CPL_SPECIFICATION.md",
        "docs/ARCHITECTURE.md",
        "src/backend/lib/modelRegistry.mo",
    ]
    for relative in required:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"canonical material for {relative}\n", encoding="utf-8")
    law = root / "cpl/laws/LAW_01.cpl"
    law.parent.mkdir(parents=True, exist_ok=True)
    law.write_text("api_key=SHOULD_BE_REDACTED_123456 doctrine law\n", encoding="utf-8")
    contract = {
        "schema": "sovereign.training.contract.v1",
        "contract_id": "freddycreates.sovereign.training.v1",
        "repository": "FreddyCreates/sovereign",
        "attribution": {"creator": "Alfredo Medina Hernandez", "required": True},
        "required_files": required,
        "include_globs": ["AGENTS.md", "cpl/laws/*.cpl"],
        "exclude_parts": [".git"],
    }
    path = root / "integration/training-contract.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract), encoding="utf-8")
    return root


def test_binding_emits_provenance_and_redacts(tmp_path: Path) -> None:
    binding = bind_sovereign(_fixture(tmp_path))
    assert binding is not None
    assert binding.receipt()["repository"] == "FreddyCreates/sovereign"
    assert isinstance(binding.receipt()["dirty"], bool)
    assert binding.receipt()["records"] == 2
    assert binding.redactions == 1
    assert "SHOULD_BE_REDACTED" not in "\n".join(binding.training_blocks())
    assert all(item["sha256"] for item in binding.receipt()["files"])


def test_binding_rejects_wrong_repository(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    contract_path = root / "integration/training-contract.v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["repository"] = "someone/else"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected Sovereign repository"):
        bind_sovereign(root)


def test_optional_missing_binding() -> None:
    assert bind_sovereign("Z:/definitely/missing/sovereign", required=False) is None


def _mock_git(monkeypatch: pytest.MonkeyPatch, *, commit: str, remote: str, dirty: bool) -> None:
    def value(_root: Path, *args: str) -> str | None:
        if args == ("rev-parse", "HEAD"):
            return commit
        if args == ("remote", "get-url", "origin"):
            return remote
        if args == ("status", "--porcelain"):
            return " M doctrine.md" if dirty else None
        return None
    monkeypatch.setattr(contract_module, "_git_value", value)


def test_production_admission_requires_exact_clean_expected_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    _mock_git(
        monkeypatch,
        commit=commit,
        remote="git@github.com:FreddyCreates/sovereign.git",
        dirty=False,
    )
    binding = bind_sovereign(
        _fixture(tmp_path),
        expected_commit=commit,
        require_clean=True,
        require_expected_remote=True,
    )
    assert binding is not None
    assert binding.receipt()["admission"]["production_admitted"] is True
    assert binding.receipt()["admission"]["remote_verified"] is True


def test_production_admission_rejects_dirty_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "b" * 40
    _mock_git(
        monkeypatch,
        commit=commit,
        remote="https://github.com/FreddyCreates/sovereign.git",
        dirty=True,
    )
    with pytest.raises(ValueError, match="dirty"):
        bind_sovereign(
            _fixture(tmp_path),
            expected_commit=commit,
            require_clean=True,
            require_expected_remote=True,
        )


def test_production_admission_rejects_commit_or_remote_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "c" * 40
    _mock_git(monkeypatch, commit=commit, remote="https://example.com/fork", dirty=False)
    with pytest.raises(ValueError, match="commit mismatch"):
        bind_sovereign(_fixture(tmp_path), expected_commit="d" * 40)
    with pytest.raises(ValueError, match="Unexpected Sovereign origin"):
        bind_sovereign(
            _fixture(tmp_path),
            expected_commit=commit,
            require_expected_remote=True,
        )
