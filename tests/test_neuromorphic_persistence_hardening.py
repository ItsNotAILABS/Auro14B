from __future__ import annotations

import hashlib
import json

from auro_native_llm.brain.feline_neuromorphic import FelineNeuromorphicEngine
from auro_native_llm.brain.neuromorphic_state import NeuromorphicStateStore
from auro_native_llm.brain.timing_plasticity import TimingPlasticityController


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _engine_and_timing():
    regions = ("V1", "V2V3", "SC", "THL_L", "LC")
    return FelineNeuromorphicEngine(regions), TimingPlasticityController(regions)


def test_persistence_restores_engine_and_timing_state(tmp_path):
    path = tmp_path / "brain.neuromorphic.json"
    store = NeuromorphicStateStore(path)
    engine, timing = _engine_and_timing()

    engine.cycle({"V1": 1.0, "SC": 0.9}, salience=0.8, novelty=0.9)
    timing.last_spike_cycle["V1"] = 7
    timing.last_spike_cycle["SC"] = 8
    saved = store.save(engine, timing)
    assert saved["durable_atomic_write"] is True

    restored_engine, restored_timing = _engine_and_timing()
    assert store.load(restored_engine, restored_timing) is True
    assert restored_engine.cycle_number == engine.cycle_number
    assert restored_engine.total_energy_ceu == engine.total_energy_ceu
    assert restored_timing.last_spike_cycle["V1"] == 7
    assert restored_timing.last_spike_cycle["SC"] == 8


def test_corrupt_state_is_quarantined_without_live_state_mutation(tmp_path):
    path = tmp_path / "brain.neuromorphic.json"
    path.write_text("{not-json", encoding="utf-8")
    store = NeuromorphicStateStore(path)
    engine, timing = _engine_and_timing()
    before_membrane = dict(engine.membrane)
    before_threshold = dict(engine.threshold)

    assert store.load(engine, timing) is False
    assert engine.membrane == before_membrane
    assert engine.threshold == before_threshold
    status = store.status()
    assert status["last_error"]
    assert status["quarantined_path"]
    assert not path.exists()


def test_hash_valid_but_malformed_state_is_transactionally_rejected(tmp_path):
    path = tmp_path / "brain.neuromorphic.json"
    store = NeuromorphicStateStore(path)
    source, timing = _engine_and_timing()
    store.save(source, timing)

    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("state_sha256")
    body["membrane"].pop("V1")
    body["state_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")

    target, target_timing = _engine_and_timing()
    target.membrane["V1"] = 0.123
    before = dict(target.membrane)
    assert store.load(target, target_timing) is False
    assert target.membrane == before
    assert store.status()["quarantined_path"]


def test_tampered_digest_is_rejected_and_quarantined(tmp_path):
    path = tmp_path / "brain.neuromorphic.json"
    store = NeuromorphicStateStore(path)
    engine, timing = _engine_and_timing()
    store.save(engine, timing)

    body = json.loads(path.read_text(encoding="utf-8"))
    body["cycle"] = body["cycle"] + 100
    path.write_text(json.dumps(body), encoding="utf-8")

    target, target_timing = _engine_and_timing()
    assert store.load(target, target_timing) is False
    assert target.cycle_number == 0
    assert store.status()["last_error"].startswith("ValueError:")
