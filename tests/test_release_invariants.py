import ast
import json
from pathlib import Path

import pytest

from auro_native_llm.open_weights import (
    CheckpointIntegrityError,
    OpenHIM,
    corpus_digest,
)


def test_production_runtime_has_one_canonical_prompt_and_single_pass_sources():
    path = Path('auro_native_llm/production_fleet/runtime.py')
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    definitions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert sum(node.name == '_agent_prompt' for node in definitions) == 1
    assert source.count('brain_cycle = self.capabilities.brain.cycle') == 1
    assert source.count('council = self.agents.run') == 1
    assert source.count('"role": "user"') == 2  # one agent call, one synthesis call


def test_runtime_imports_canonically():
    from auro_native_llm.production_fleet.runtime import NovaRuntime, AgentManager
    assert NovaRuntime is not None
    assert AgentManager is not None


def test_open_weight_loader_rehashes_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv('AURO_REQUIRE_SIGNED_TRAINING_RECEIPT', '1')
    model = OpenHIM()
    key = 'runner-secret'
    model.save(
        tmp_path,
        {'train_loss': 0.0776, 'held_out_loss': 3.5374, 'perplexity': 34.38},
        corpus_sha256=corpus_digest('tiny corpus'),
        source_commit='deadbeef',
        runner_identity='ci-runner:test',
        signing_key=key,
    )
    loaded = OpenHIM.load(tmp_path, runner_signing_key=key)
    assert loaded.verified_checkpoint['verified'] is True
    assert loaded.verified_checkpoint['claim_class'] == 'pipeline-fixture-only'
    (tmp_path / 'weights.npz.b64').write_text('tampered', encoding='ascii')
    with pytest.raises(CheckpointIntegrityError):
        OpenHIM.load(tmp_path, runner_signing_key=key)


def test_training_receipt_requires_immutable_provenance(tmp_path):
    model = OpenHIM()
    with pytest.raises(ValueError):
        model.save(tmp_path, {}, corpus_sha256='', source_commit='', runner_identity='')
