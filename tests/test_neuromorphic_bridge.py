from auro_native_llm.production_fleet.neuromorphic_bridge import (
    NeuromorphicAwareGenerator,
    compact_neuromorphic_state,
    neuromorphic_model_preferences,
)


class FakeBrain:
    def __init__(self, *, pressure=0.62, orienting=True, spike_rate=0.125):
        self.cycle = 3
        self.pressure = pressure
        self.orienting = orienting
        self.spike_rate = spike_rate

    def snapshot(self):
        return {
            "neuromorphic": {"cycle": self.cycle},
            "last_neuromorphic_cycle": {
                "cycle": self.cycle,
                "spike_rate": self.spike_rate,
                "sparsity": 1.0 - self.spike_rate,
                "inhibitory_tone": 0.2,
                "synaptic_events": 9,
                "energy_pressure": self.pressure,
                "orienting_burst": self.orienting,
                "active_regions": ["SC", "V1"],
            },
        }


class FakeOrchestrator:
    def __init__(self):
        self.preferences = ()
        self.messages = None

    def set_preferred_models(self, preferences):
        self.preferences = tuple(preferences)

    def __call__(self, messages, options):
        self.messages = messages
        return {"text": "ok"}


def test_compact_state_is_bounded_and_non_authoritative():
    state = compact_neuromorphic_state(FakeBrain())
    assert state["cycle"] == 3
    assert state["spike_rate"] == 0.125
    assert state["active_regions"] == ["SC", "V1"]
    assert state["authority"] == "telemetry_only"
    assert state["can_authorize_execution"] is False


def test_energy_pressure_prefers_smaller_model_lanes_first():
    state = compact_neuromorphic_state(FakeBrain(pressure=1.4, orienting=False))
    prefs = neuromorphic_model_preferences(state, ("AURO-ST-14B", "Auro-14B", "Auro-8B"))
    assert prefs[:2] == ("Auro-2B", "Auro-4B")
    assert "AURO-ST-14B" in prefs


def test_orienting_burst_prefers_fast_edge_specialist_lanes():
    state = compact_neuromorphic_state(FakeBrain(pressure=0.4, orienting=True))
    prefs = neuromorphic_model_preferences(state, ("Auro-8B", "Auro-14B"))
    assert prefs[:3] == ("Auro-4B", "Auro-2B", "AURO-ST-14B")


def test_generator_injects_telemetry_after_authoritative_system_message_and_routes():
    orchestrator = FakeOrchestrator()
    original = [
        {"role": "system", "content": "authoritative persona instruction"},
        {"role": "user", "content": "hello"},
    ]
    wrapped = NeuromorphicAwareGenerator(
        orchestrator,
        lambda: FakeBrain(pressure=1.2, orienting=False),
        base_preferences=("Auro-8B", "Auro-14B"),
    )
    result = wrapped(original, {"temperature": 0.2})

    assert original[0]["content"] == "authoritative persona instruction"
    assert orchestrator.messages[0] == original[0]
    assert orchestrator.messages[1]["role"] == "system"
    assert "HIM_NEUROMORPHIC_STATE" in orchestrator.messages[1]["content"]
    assert "not execution approval" in orchestrator.messages[1]["content"]
    assert orchestrator.messages[2] == original[1]
    assert orchestrator.preferences[:2] == ("Auro-2B", "Auro-4B")
    assert result["neuromorphic_context"]["can_authorize_execution"] is False
    assert result["neuromorphic_context"]["routing_preferences"][:2] == ["Auro-2B", "Auro-4B"]
