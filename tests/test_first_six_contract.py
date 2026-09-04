"""Contract tests for the first six local AURO lanes.

These tests require only the standard library plus the repository package.
They verify the family registry and deterministic organism benchmark; they do
not require checkpoint weights or a network service.
"""

from __future__ import annotations

from auro_native_llm.model.config import family_config
from benchmarks.sovereign_organism_trial import make_episode, run_condition


FIRST_SIX = (
    "Auro-156K",
    "Auro-320M",
    "Auro-640M",
    "Auro-1B",
    "Auro-2B",
    "Auro-3B",
)


def test_first_six_have_local_dev_profiles() -> None:
    for model_id in FIRST_SIX:
        config = family_config(model_id, mode="dev")
        assert config.model_id == model_id
        assert config.mode == "dev"
        assert config.max_seq_len > 0
        assert config.hidden_dim > 0


def test_organism_trial_is_deterministic() -> None:
    episodes = [make_episode(i, 23) for i in range(12)]
    first = run_condition("organism", episodes)
    second = run_condition("organism", episodes)
    assert [r.receipt_replay_equal for r in first] == [True] * len(first)
    assert [(r.action, r.success) for r in first] == [
        (r.action, r.success) for r in second
    ]
