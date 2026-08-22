from auro_native_llm.production_fleet.neuromorphic_bridge import (
    NeuromorphicAwareGenerator,
    compact_neuromorphic_state,
)


class FakeBrain:
    def __init__(self):
        self.cycle = 3

    def snapshot(self):
        return {
            "neuromorphic": {"cycle": self.cycle},
            "last_neuromorphic_cycle": {
                "cycle": self.cycle,
                "spike_rate": 0.125,
                "sparsity": 0.875,
                "inhibitory_tone": 0.2,
                "synaptic_events": 9,
                "energy_pressure": 0.62,
                "orienting_burst": True,
                "active_regions": ["SC", "V1"],
            },
        }


def test_compact_state_is_bounded_and_non_authoritative():
    state = compact_neuromorphic_state(FakeBrain())
    assert state["cycle"] == 3
    assert state["spike_rate"] == 0.125
    assert state["active_regions"] == ["SC", "V1"]
    assert state["authority"] == "telemetry_only"
    assert state["can_authorize_execution"] is False


def test_generator_injects_neuromorphic_state_without_mutating_input_messages():
    seen = {}

    def generator(messages, options):
        seen["messages"] = messages
        seen["options"] = options
        return {"text": "ok"}

    original = [{"role": "user", "content": "hello"}]
    wrapped = NeuromorphicAwareGenerator(generator, FakeBrain)
    result = wrapped(original, {"temperature": 0.2})

    assert original == [{"role": "user", "content": "hello"}]
    assert seen["messages"][0]["role"] == "system"
    assert "HIM_NEUROMORPHIC_STATE" in seen["messages"][0]["content"]
    assert "not execution approval" in seen["messages"][0]["content"]
    assert seen["messages"][1] == original[0]
    assert result["neuromorphic_context"]["can_authorize_execution"] is False
