from types import SimpleNamespace

import numpy as np

from auro_native_llm.model.family_upgrade import apply_family_upgrade
from auro_native_llm.neuro import NeuroBridge, SpikingGateConfig, SpikingResidualGate


def test_spiking_gate_is_sparse_bounded_and_emits_regularizer():
    gate = SpikingResidualGate(SpikingGateConfig(target_activity=0.18, threshold_quantile=0.82))
    hidden = np.linspace(-2.0, 2.0, 32, dtype=np.float64).reshape(1, 1, 32)
    residual = np.ones(32, dtype=np.float64) * 0.25
    gated, receipt = gate.apply(hidden, residual)
    assert gated.shape == residual.shape
    assert 0.0 <= receipt.activity_rate <= 1.0
    assert 0.0 <= receipt.sparsity <= 1.0
    assert 0.0 <= receipt.gate_mean <= 1.0
    assert receipt.regularizer >= 0.0
    assert receipt.physical_energy_claim is False
    assert receipt.biological_equivalence_claim is False
    assert np.all(np.abs(gated) <= np.abs(residual) + 1e-12)


def test_family_upgrade_installs_neuromorphic_policy_without_double_context_growth():
    config = SimpleNamespace(
        max_seq_len=512,
        use_moe=False,
        num_experts=1,
        top_k_experts=1,
        moe_every=2,
        extra={},
    )
    apply_family_upgrade(config)
    first_context = config.max_seq_len
    apply_family_upgrade(config)
    assert first_context == 2048
    assert config.max_seq_len == first_context
    assert config.extra["use_neuromorphic_residual"] is True
    assert config.extra["neuromorphic_residual_policy"] == "auro.family.neuromorphic-residual.v1"
    assert config.extra["neuromorphic_checkpoint_quality_verified"] is False
    assert config.extra["neuromorphic_physical_energy_verified"] is False


class FakeCore:
    def __init__(self, dim):
        self.lm_head_weight = np.ones((dim, 7), dtype=np.float64) * 0.01


class FakeLanguage:
    def __init__(self, enabled: bool, dim: int = 16):
        self.config = SimpleNamespace(
            hidden_dim=dim,
            num_heads=2,
            extra={
                "use_neuromorphic_residual": enabled,
                "neuromorphic_target_activity": 0.18,
                "neuromorphic_threshold_quantile": 0.82,
                "neuromorphic_minimum_gate": 0.35,
                "neuromorphic_energy_penalty_weight": 0.01,
                "neuromorphic_activity_penalty_weight": 0.02,
                "neuromorphic_inhibitory_gain": 0.35,
            },
        )
        self.core = FakeCore(dim)
        self._neuro = None


def test_neurobridge_disabled_path_preserves_normal_neuroemergence_contract():
    language = FakeLanguage(False)
    bridge = NeuroBridge(language)
    hidden = np.ones((1, 3, 16), dtype=np.float64) * 0.1
    outputs = bridge.fuse_forward_outputs({"last_hidden_state": hidden.copy()}, text="visual signal")
    assert outputs["neuromorphic_residual_enabled"] is False
    assert outputs["neuromorphic_residual"] is None
    assert outputs["neuromorphic_regularizer"] == 0.0
    assert outputs["last_hidden_state"].shape == hidden.shape
    assert outputs["logits"].shape == (1, 3, 7)


def test_neurobridge_enabled_path_gates_only_last_token_delta():
    language = FakeLanguage(True)
    bridge = NeuroBridge(language)
    hidden = np.linspace(-0.2, 0.2, 48, dtype=np.float64).reshape(1, 3, 16)
    outputs = bridge.fuse_forward_outputs({"last_hidden_state": hidden.copy()}, text="visual orienting signal")
    fused = outputs["last_hidden_state"]
    assert outputs["neuromorphic_residual_enabled"] is True
    assert outputs["neuromorphic_residual"] is not None
    assert outputs["neuromorphic_regularizer"] >= 0.0
    assert np.allclose(fused[:, :-1, :], hidden[:, :-1, :])
    assert fused[:, -1, :].shape == hidden[:, -1, :].shape
    assert outputs["neuromorphic_residual"]["physical_energy_claim"] is False


def test_neurobridge_reports_unverified_quality_and_energy_boundaries():
    info = NeuroBridge(FakeLanguage(True)).info()
    assert info["neuromorphic_residual_enabled"] is True
    assert info["spiking_gate"]["trainable_parameters"] == 0
    assert info["checkpoint_quality_verified"] is False
    assert info["physical_energy_efficiency_verified"] is False
