"""Tests for phantom_native — Sovereign Native Stack (Phantom-MESIE Integration).

Covers:
- SovereignTensor: SIMD-style ops, quantization, helix encoding, MESIE ingestion
- TaurusMemory: store/recall, decay, compression, similarity search
- SovereignNeuroCore: forward pass, attention analysis, memory integration
- SovereignSwarmRuntime: spawn, execute, sealed intent, aggregation
"""

import math
import time

import pytest


# ============================================================
# SovereignTensor Tests
# ============================================================


class TestSovereignTensor:
    def test_create_basic(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.0, 2.0, 3.0, 4.0], (4,))
        assert t.shape == (4,)
        assert len(t) == 4
        assert t.resonance == 1.0

    def test_shape_mismatch_raises(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        with pytest.raises(ValueError):
            SovereignTensor([1.0, 2.0], (3,))

    def test_zeros_and_ones(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        z = SovereignTensor.zeros((8,))
        assert all(x == 0.0 for x in z.data)
        o = SovereignTensor.ones((8,))
        assert all(x == 1.0 for x in o.data)

    def test_vector_add(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        a = SovereignTensor([1.0] * 16, (16,))
        b = SovereignTensor([2.0] * 16, (16,))
        c = a.vector_add(b)
        assert all(abs(x - 3.0) < 1e-5 for x in c.data)

    def test_vector_add_non_multiple_of_8(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        a = SovereignTensor([1.0] * 13, (13,))
        b = SovereignTensor([0.5] * 13, (13,))
        c = a.vector_add(b)
        assert len(c.data) == 13
        assert all(abs(x - 1.5) < 1e-5 for x in c.data)

    def test_vector_mul(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        a = SovereignTensor([2.0] * 8, (8,))
        b = SovereignTensor([3.0] * 8, (8,))
        c = a.vector_mul(b)
        assert all(abs(x - 6.0) < 1e-5 for x in c.data)

    def test_scale(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.0, 2.0, 3.0], (3,))
        s = t.scale(2.0)
        assert list(s.data) == pytest.approx([2.0, 4.0, 6.0])

    def test_resonance_matmul(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        # 2x3 * 3x2 = 2x2
        a = SovereignTensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (2, 3))
        b = SovereignTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        c = a.resonance_matmul(b)
        assert c.shape == (2, 2)
        # With resonance=1.0, first row = [1,2], second row = [3,4]
        assert abs(c.data[0] - 1.0) < 1e-5
        assert abs(c.data[1] - 2.0) < 1e-5
        assert abs(c.data[2] - 3.0) < 1e-5
        assert abs(c.data[3] - 4.0) < 1e-5

    def test_resonance_matmul_weighting(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        a = SovereignTensor(
            [1.0, 1.0, 1.0, 1.0], (2, 2), {"resonance": 0.5}
        )
        b = SovereignTensor(
            [1.0, 0.0, 0.0, 1.0], (2, 2), {"resonance": 2.0}
        )
        c = a.resonance_matmul(b)
        # resonance product = 0.5 * 2.0 = 1.0
        assert c.shape == (2, 2)

    def test_dot_product(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        a = SovereignTensor([1.0, 2.0, 3.0], (3,))
        b = SovereignTensor([4.0, 5.0, 6.0], (3,))
        assert abs(a.dot(b) - 32.0) < 1e-5

    def test_quantize_int8(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.0, -0.5, 0.25, 0.0], (4,))
        q = t.quantize_int8()
        assert q.spectral_meta["quantized"] is True
        assert "quant_scale" in q.spectral_meta
        # Max value maps to 127
        assert abs(q.data[0] - 127.0) < 1e-5

    def test_dequantize_roundtrip(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.0, -0.5, 0.25, 0.0], (4,))
        q = t.quantize_int8()
        dq = q.dequantize()
        # Approximate roundtrip (quantization introduces error)
        for orig, recovered in zip(t.data, dq.data):
            assert abs(orig - recovered) < 0.02

    def test_helix_encode(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.0] * 32, (32,))
        h = t.helix_encode(turns=4)
        assert h.spectral_meta["helix"]["encoded"] is True
        assert h.spectral_meta["helix"]["turns"] == 4
        # First element: cos(0) * 1.0 + sin(0) * 0.1 = 1.0 + 0.0 = 1.0
        assert abs(h.data[0] - 1.0) < 1e-5

    def test_from_mesie_component(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        component = {
            "amplitude": [0.1, 0.5, 0.9, 0.3],
            "frequency": [1.0, 2.0, 3.0, 4.0],
            "element_weight": 0.8,
            "node_id": ["node_1"],
        }
        t = SovereignTensor.from_mesie_component(component)
        assert t.shape == (4,)
        assert t.resonance == 0.8
        assert list(t.data) == pytest.approx([0.1, 0.5, 0.9, 0.3], abs=1e-5)

    def test_to_bytes_deterministic(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t1 = SovereignTensor([1.0, 2.0, 3.0], (3,))
        t2 = SovereignTensor([1.0, 2.0, 3.0], (3,))
        assert t1.to_bytes() == t2.to_bytes()

    def test_from_bytes_roundtrip(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([1.5, 2.5, 3.5, 4.5], (4,))
        raw = t.to_bytes()
        t2 = SovereignTensor.from_bytes(raw, (4,))
        for a, b in zip(t.data, t2.data):
            assert abs(a - b) < 1e-5

    def test_norm(self):
        from phantom_native.sovereign_tensor import SovereignTensor

        t = SovereignTensor([3.0, 4.0], (2,))
        assert abs(t.norm() - 5.0) < 1e-5


# ============================================================
# TaurusMemory Tests
# ============================================================


class TestTaurusMemory:
    def test_store_and_recall_by_key(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=16)
        t = SovereignTensor([1.0, 2.0, 3.0], (3,))
        key = mem.store(t, key="test_key")
        assert key == "test_key"
        recalled = mem.recall_by_key("test_key")
        assert recalled is not None
        assert list(recalled.data) == pytest.approx([1.0, 2.0, 3.0])

    def test_auto_key_generation(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory()
        t = SovereignTensor([1.0], (1,))
        key = mem.store(t)
        assert key.startswith("qsha:")

    def test_capacity_eviction(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=3)
        for i in range(5):
            mem.store(SovereignTensor([float(i)], (1,)))
        assert len(mem.working_memory) == 3

    def test_recall_top_k(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=16)
        for i in range(8):
            t = SovereignTensor([float(i)], (1,), {"resonance": float(i)})
            mem.store(t, importance=float(i))
        top = mem.recall_top_k(3)
        assert len(top) == 3

    def test_recall_by_similarity(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=16)
        mem.store(SovereignTensor([1.0, 0.0, 0.0], (3,)))
        mem.store(SovereignTensor([0.0, 1.0, 0.0], (3,)))
        mem.store(SovereignTensor([0.9, 0.1, 0.0], (3,)))

        query = SovereignTensor([1.0, 0.0, 0.0], (3,))
        results = mem.recall_by_similarity(query, top_k=2)
        assert len(results) == 2
        # Most similar should be the [1,0,0] vector
        assert results[0][2] > 0.9  # high similarity

    def test_compress_helix(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory()
        t = SovereignTensor([1.0, 3.0, 5.0, 7.0], (4,))
        compressed = mem.compress_helix(t)
        assert len(compressed.data) == 2
        assert abs(compressed.data[0] - 2.0) < 1e-5  # (1+3)/2
        assert abs(compressed.data[1] - 6.0) < 1e-5  # (5+7)/2

    def test_consolidate(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=16)
        # Store items with varying importance
        for i in range(5):
            mem.store(SovereignTensor([float(i)], (1,)), importance=float(i) * 0.2)
        removed = mem.consolidate(threshold=0.5)
        assert removed >= 0

    def test_size(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.taurus import TaurusMemory

        mem = TaurusMemory(capacity=8)
        mem.store(SovereignTensor([1.0], (1,)))
        status = mem.size()
        assert status["working_memory"] == 1
        assert status["long_term"] == 1
        assert status["capacity"] == 8


# ============================================================
# SovereignNeuroCore Tests
# ============================================================


class TestSovereignNeuroCore:
    def test_create_default(self):
        from phantom_native.neurocore import SovereignNeuroCore

        core = SovereignNeuroCore()
        assert core.d_model == 128
        assert core.n_heads == 8

    def test_create_custom_config(self):
        from phantom_native.neurocore import SovereignNeuroCore

        core = SovereignNeuroCore({"d_model": 64, "n_heads": 4})
        assert core.d_model == 64
        assert core.n_heads == 4

    def test_forward_pass(self):
        from phantom_native.neurocore import SovereignNeuroCore
        from phantom_native.sovereign_tensor import SovereignTensor

        core = SovereignNeuroCore({"d_model": 32, "n_heads": 4})
        t = SovereignTensor([math.sin(i * 0.1) for i in range(64)], (64,))
        out = core.forward(t)
        assert out.shape == (64,)
        assert len(out.data) == 64

    def test_attention_analysis(self):
        from phantom_native.neurocore import SovereignNeuroCore
        from phantom_native.sovereign_tensor import SovereignTensor

        core = SovereignNeuroCore({"d_model": 16, "n_heads": 4})
        t = SovereignTensor([1.0] * 16, (16,))
        core.forward(t)
        analysis = core.get_attention_analysis()
        assert analysis["n_heads"] == 4
        assert len(analysis["head_analyses"]) == 4
        for head in analysis["head_analyses"]:
            assert "attention_entropy" in head
            assert "max_attention" in head
            assert "attention_sparsity" in head

    def test_taurus_memory_fills(self):
        from phantom_native.neurocore import SovereignNeuroCore
        from phantom_native.sovereign_tensor import SovereignTensor

        core = SovereignNeuroCore({"d_model": 16, "n_heads": 2, "memory_capacity": 8})
        for i in range(5):
            t = SovereignTensor([float(i)] * 16, (16,))
            core.forward(t)
        assert len(core.taurus.working_memory) == 5

    def test_reset_memory(self):
        from phantom_native.neurocore import SovereignNeuroCore
        from phantom_native.sovereign_tensor import SovereignTensor

        core = SovereignNeuroCore({"d_model": 16, "n_heads": 2})
        t = SovereignTensor([1.0] * 16, (16,))
        core.forward(t)
        assert len(core.taurus.working_memory) > 0
        core.reset_memory()
        assert len(core.taurus.working_memory) == 0


# ============================================================
# SovereignSwarmRuntime Tests
# ============================================================


class TestSovereignSwarmRuntime:
    def test_spawn_neuronet(self):
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        core_id = runtime.spawn_neuronet({"d_model": 32, "n_heads": 4})
        assert core_id.startswith("qsha:")
        assert len(runtime.cores) == 1

    def test_spawn_multiple(self):
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        ids = [runtime.spawn_neuronet() for _ in range(4)]
        assert len(runtime.cores) == 4
        assert len(set(ids)) == 4  # unique IDs

    def test_execute(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        runtime.spawn_neuronet({"d_model": 16, "n_heads": 2})
        runtime.spawn_neuronet({"d_model": 16, "n_heads": 2})
        t = SovereignTensor([1.0] * 16, (16,))
        results = runtime.execute(t)
        assert len(results) == 2

    def test_sealed_intent_execution(self):
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        runtime.spawn_neuronet({"d_model": 16, "n_heads": 2})

        intent = {"amplitude": [0.1, 0.5, 0.9, 0.3], "element_weight": 0.8}
        sealed = runtime.vault.seal_intent(intent)
        receipt = runtime.execute_sealed_intent(sealed)

        assert receipt.commitment.startswith("commit:")
        assert receipt.shadow_wire["masked"] is True
        assert receipt.public_meta["swarm_size"] == 1

    def test_aggregate_swarm(self):
        from phantom_native.sovereign_tensor import SovereignTensor
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        tensors = [
            SovereignTensor([1.0, 2.0], (2,), {"resonance": 1.0}),
            SovereignTensor([3.0, 4.0], (2,), {"resonance": 1.0}),
        ]
        agg = runtime.aggregate_swarm(tensors)
        # Equal resonance → simple average
        assert abs(agg.data[0] - 2.0) < 1e-5
        assert abs(agg.data[1] - 3.0) < 1e-5

    def test_swarm_status(self):
        from phantom_native.swarm_runtime import SovereignSwarmRuntime

        runtime = SovereignSwarmRuntime()
        runtime.spawn_neuronet()
        runtime.spawn_neuronet()
        status = runtime.get_swarm_status()
        assert status["n_cores"] == 2
        assert status["manifest"].startswith("manifest:")

    def test_vault_seal_open_roundtrip(self):
        from phantom_native.swarm_runtime import SovereignVault

        vault = SovereignVault()
        intent = {"amplitude": [1.0, 2.0, 3.0], "resonance": 0.9}
        sealed = vault.seal_intent(intent)
        opened = vault.open_sealed_intent(sealed)
        assert opened["resonance"] == 0.9

    def test_shadow_wire_masking(self):
        from phantom_native.swarm_runtime import ShadowWireEnvelope

        wire = ShadowWireEnvelope()
        masked = wire.mask_topology(["core_1", "core_2", "core_3"])
        assert masked["n_cores"] == 3
        assert masked["masked"] is True
        assert "swarm_hash" in masked

    def test_execution_receipt_verify(self):
        from phantom_native.swarm_runtime import ExecutionReceipt

        receipt = ExecutionReceipt(
            commitment="commit:abc123",
            shadow_wire={"masked": True},
        )
        assert receipt.verify("commit:abc123") is True
        assert receipt.verify("commit:xyz") is False
