"""Tests for MESIE verification engines — hardware abstraction, scalability,
attestation, reproducibility, and auto-validation agent.

These tests address the review concerns:
- Hardware-in-the-loop readiness
- Scalability to 10K+ nodes
- Cryptographic attestation of execution
- Independent reproducibility verification
"""

from __future__ import annotations

import numpy as np
import pytest

from mesie.engines.hardware_abstraction_engine import (
    DeviceClass,
    DeviceDescriptor,
    DeviceState,
    HardwareAbstractionEngine,
    SensorReading,
)
from mesie.engines.scalability_engine import ScalabilityEngine, ScalabilityResult
from mesie.engines.attestation_engine import (
    AttestationChain,
    AttestationEngine,
    ExecutionAttestation,
)
from mesie.engines.reproducibility_engine import ReproducibilityEngine, ReproducibilityProof
from mesie.engines.auto_validation_agent import AutoValidationAgent, run_auto_validation


# ===========================================================================
# Hardware Abstraction Engine Tests
# ===========================================================================


class TestHardwareAbstractionEngine:
    """Tests for the Hardware Abstraction Layer engine."""

    def test_create_simulated_device(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(
            device_class=DeviceClass.ACCELEROMETER,
            sample_rate_hz=1000.0,
            resolution_bits=16,
        )
        assert device.device_class == DeviceClass.ACCELEROMETER
        assert device.sample_rate_hz == 1000.0
        assert device.state == DeviceState.READY
        assert len(device.calibration_hash) > 0

    def test_register_multiple_device_classes(self):
        hal = HardwareAbstractionEngine()
        classes = [DeviceClass.VIBRATION_SENSOR, DeviceClass.IMU, DeviceClass.MICROPHONE]
        for dc in classes:
            device = hal.create_simulated_device(device_class=dc)
            assert device.device_class == dc
            assert device.state == DeviceState.READY

    def test_read_sensor(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.ACCELEROMETER)
        reading = hal.read_sensor(device.device_id, n_samples=256, seed=42)
        assert isinstance(reading, SensorReading)
        assert len(reading.values) == 256
        assert reading.is_simulated is True
        assert reading.device_id == device.device_id
        assert len(reading.integrity_hash) > 0

    def test_sensor_reading_deterministic(self):
        """Same seed produces same reading."""
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.ACCELEROMETER)
        r1 = hal.read_sensor(device.device_id, n_samples=128, seed=42)
        r2 = hal.read_sensor(device.device_id, n_samples=128, seed=42)
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_stream_burst(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.VIBRATION_SENSOR)
        readings = hal.stream_burst(device.device_id, n_readings=10, seed=42)
        assert len(readings) == 10
        for r in readings:
            assert len(r.values) == 256

    def test_fault_injection_disconnect(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.ACCELEROMETER)
        result = hal.inject_fault(device.device_id, "disconnect")
        assert result["handled"] is True
        assert device.state == DeviceState.OFFLINE

    def test_fault_injection_noise_spike(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.ACCELEROMETER)
        result = hal.inject_fault(device.device_id, "noise_spike")
        assert result["handled"] is True
        assert device.state == DeviceState.DEGRADED

    def test_read_after_fault_raises(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.ACCELEROMETER)
        hal.inject_fault(device.device_id, "disconnect")
        with pytest.raises(RuntimeError):
            hal.read_sensor(device.device_id)

    def test_verify_timing_within_budget(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(DeviceClass.SPECTRAL_ANALYZER)
        # Generate some readings to populate timing log
        for i in range(20):
            hal.read_sensor(device.device_id, n_samples=256, seed=i)
        timing = hal.verify_timing(budget_ms=100.0)
        assert timing["all_within_budget"] is True
        assert timing["n_readings"] == 20

    def test_validate_interface_contract(self):
        hal = HardwareAbstractionEngine()
        device = hal.create_simulated_device(
            DeviceClass.ACCELEROMETER,
            sample_rate_hz=1000.0,
            resolution_bits=16,
            frequency_range=(0.1, 500.0),
        )
        result = hal.validate_interface_contract(device.device_id)
        assert result["contract_valid"] is True
        assert result["n_passed"] == result["n_total"]

    def test_contract_nyquist_violation(self):
        """Device with frequency range exceeding Nyquist should fail."""
        hal = HardwareAbstractionEngine()
        # Manually create a device that violates Nyquist
        device = DeviceDescriptor(
            device_id="bad-device",
            device_class=DeviceClass.ACCELEROMETER,
            sample_rate_hz=100.0,  # Nyquist = 50 Hz
            resolution_bits=16,
            frequency_range=(0.1, 200.0),  # Exceeds Nyquist!
            amplitude_range=(-32768, 32767),
            latency_ms=2.0,
            jitter_ms=0.5,
            state=DeviceState.READY,
        )
        hal._devices[device.device_id] = device
        result = hal.validate_interface_contract(device.device_id)
        assert result["contract_valid"] is False

    def test_handle_bus_message_register(self):
        from mesie.internal_api.messages import MessageEnvelope, MessageTopic

        hal = HardwareAbstractionEngine()
        msg = MessageEnvelope(
            topic=MessageTopic.ENGINE_REQUEST,
            source="test",
            target="hardware_abstraction",
            action="register_device",
            payload={"device_class": "accelerometer", "sample_rate_hz": 2000.0},
        )
        resp = hal.handle(msg)
        assert resp is not None
        assert resp.ok is True
        assert "device_id" in resp.data


# ===========================================================================
# Scalability Engine Tests
# ===========================================================================


class TestScalabilityEngine:
    """Tests for the scalability stress-testing engine."""

    def test_spawn_network(self):
        engine = ScalabilityEngine()
        net_id, nodes = engine.spawn_network(100, seed=42)
        assert len(nodes) == 100
        assert all(n.health > 0 for n in nodes)

    def test_spawn_large_network(self):
        """Test spawning 5000 nodes."""
        engine = ScalabilityEngine()
        net_id, nodes = engine.spawn_network(5000, seed=42)
        assert len(nodes) == 5000

    def test_stress_test_1000_nodes(self):
        engine = ScalabilityEngine()
        result = engine.stress_test(n_nodes=1000, n_operations=5000, seed=42)
        assert isinstance(result, ScalabilityResult)
        assert result.n_nodes == 1000
        assert result.all_nodes_healthy is True
        assert result.operations_per_sec > 0
        assert result.latency_mean_ms < 10.0  # Should be sub-ms

    def test_stress_test_deterministic(self):
        """Same seed produces same checksum."""
        engine = ScalabilityEngine()
        r1 = engine.stress_test(n_nodes=500, n_operations=1000, seed=42)
        r2 = engine.stress_test(n_nodes=500, n_operations=1000, seed=42)
        assert r1.checksum == r2.checksum

    def test_stress_test_10k_nodes(self):
        """Prove 10K nodes can be managed."""
        engine = ScalabilityEngine()
        result = engine.stress_test(n_nodes=10000, n_operations=10000, seed=42)
        assert result.n_nodes == 10000
        assert result.all_nodes_healthy is True
        # Should complete in reasonable time
        assert result.elapsed_s < 30.0

    def test_measure_throughput(self):
        engine = ScalabilityEngine()
        result = engine.measure_throughput(n_nodes=1000, duration_s=0.5, seed=42)
        assert result["n_nodes"] == 1000
        assert result["throughput_ops_per_sec"] > 0

    def test_scaling_analysis(self):
        engine = ScalabilityEngine()
        result = engine.scaling_analysis(node_counts=[100, 500, 1000], seed=42)
        assert len(result["results"]) == 3
        assert result["max_nodes_tested"] == 1000

    def test_verify_determinism(self):
        engine = ScalabilityEngine()
        result = engine.verify_determinism(n_nodes=200, seed=42)
        assert result["deterministic"] is True
        assert result["checksum_1"] == result["checksum_2"]

    def test_handle_bus_stress_test(self):
        from mesie.internal_api.messages import MessageEnvelope, MessageTopic

        engine = ScalabilityEngine()
        msg = MessageEnvelope(
            topic=MessageTopic.ENGINE_REQUEST,
            source="test",
            target="scalability",
            action="stress_test",
            payload={"n_nodes": 200, "n_operations": 500, "seed": 42},
        )
        resp = engine.handle(msg)
        assert resp is not None
        assert resp.ok is True
        assert resp.data["n_nodes"] == 200


# ===========================================================================
# Attestation Engine Tests
# ===========================================================================


class TestAttestationEngine:
    """Tests for the cryptographic attestation engine."""

    def test_create_chain(self):
        engine = AttestationEngine()
        chain = engine.create_chain("test-chain")
        assert chain.chain_id == "test-chain"
        assert chain.length == 0
        assert chain.is_valid is True

    def test_attest_computation(self):
        engine = AttestationEngine()
        engine.create_chain("test")
        attestation = engine.attest_computation(
            chain_id="test",
            operation="spectral_match",
            input_data={"signal": [1, 2, 3]},
            output_data={"score": 0.95},
            execution_time_ms=1.5,
            seed=42,
        )
        assert isinstance(attestation, ExecutionAttestation)
        assert attestation.operation == "spectral_match"
        assert attestation.chain_hash == "genesis"
        assert len(attestation.input_hash) == 16
        assert len(attestation.output_hash) == 16

    def test_chain_links_correctly(self):
        engine = AttestationEngine()
        engine.create_chain("chain1")
        a1 = engine.attest_computation("chain1", "op1", "in1", "out1", 1.0)
        a2 = engine.attest_computation("chain1", "op2", "in2", "out2", 2.0)
        a3 = engine.attest_computation("chain1", "op3", "in3", "out3", 3.0)

        assert a1.chain_hash == "genesis"
        assert a2.chain_hash != "genesis"
        assert a3.chain_hash != a2.chain_hash

        # Verify chain is valid
        result = engine.verify_chain("chain1")
        assert result["valid"] is True
        assert result["length"] == 3

    def test_chain_verification(self):
        engine = AttestationEngine()
        engine.create_chain("verify-me")
        for i in range(10):
            engine.attest_computation("verify-me", f"op-{i}", f"in-{i}", f"out-{i}", float(i))
        result = engine.verify_chain("verify-me")
        assert result["valid"] is True
        assert result["length"] == 10

    def test_attest_test_run(self):
        engine = AttestationEngine()
        engine.create_chain("tests")
        attestation = engine.attest_test_run(
            chain_id="tests",
            test_name="test_spectral_match",
            passed=True,
            elapsed_ms=0.5,
            assertions=3,
            seed=42,
        )
        assert "test:test_spectral_match" == attestation.operation
        assert attestation.metadata["passed"] is True

    def test_export_evidence(self):
        engine = AttestationEngine()
        engine.create_chain("export-test")
        for i in range(5):
            engine.attest_computation("export-test", f"step-{i}", f"in-{i}", f"out-{i}", 1.0)
        evidence = engine.export_evidence("export-test")
        assert evidence["valid"] is True
        assert evidence["length"] == 5
        assert len(evidence["attestations"]) == 5
        assert "platform" in evidence

    def test_platform_info_captured(self):
        engine = AttestationEngine()
        engine.create_chain("plat")
        a = engine.attest_computation("plat", "op", "in", "out", 1.0)
        assert "system" in a.platform_info
        assert "python" in a.platform_info
        assert "numpy" in a.platform_info


# ===========================================================================
# Reproducibility Engine Tests
# ===========================================================================


class TestReproducibilityEngine:
    """Tests for the reproducibility proof engine."""

    def test_generate_spectral_proof(self):
        engine = ReproducibilityEngine()
        proof = engine.generate_spectral_proof(seed=42, n_verifications=5)
        assert isinstance(proof, ReproducibilityProof)
        assert proof.all_match is True
        assert proof.n_verifications == 5
        assert proof.seed == 42
        assert len(proof.output_hash) == 16

    def test_spectral_proof_deterministic(self):
        """Same seed always produces same proof hash."""
        engine = ReproducibilityEngine()
        p1 = engine.generate_spectral_proof(seed=42)
        p2 = engine.generate_spectral_proof(seed=42)
        assert p1.output_hash == p2.output_hash

    def test_different_seeds_different_hashes(self):
        engine = ReproducibilityEngine()
        p1 = engine.generate_spectral_proof(seed=42)
        p2 = engine.generate_spectral_proof(seed=99)
        assert p1.output_hash != p2.output_hash

    def test_prove_computation_custom(self):
        engine = ReproducibilityEngine()

        def my_computation(rng):
            return rng.normal(0, 1, 100)

        proof = engine.prove_computation(
            operation="custom_normal",
            compute_fn=my_computation,
            seed=123,
            n_verifications=5,
        )
        assert proof.all_match is True
        assert proof.operation == "custom_normal"

    def test_batch_verify_100_seeds(self):
        engine = ReproducibilityEngine()
        result = engine.batch_verify(seeds=list(range(100)))
        assert result["all_reproducible"] is True
        assert result["pass_rate"] == 1.0
        assert result["n_seeds"] == 100

    def test_verification_code_generated(self):
        engine = ReproducibilityEngine()
        proof = engine.generate_spectral_proof(seed=42)
        assert "np.random.default_rng(42)" in proof.verification_code
        assert "signal" in proof.verification_code

    def test_export_proofs(self):
        engine = ReproducibilityEngine()
        for seed in range(10):
            engine.generate_spectral_proof(seed=seed)
        export = engine.export_proofs()
        assert export["n_proofs"] == 10
        assert export["all_reproducible"] is True


# ===========================================================================
# Auto-Validation Agent Tests
# ===========================================================================


class TestAutoValidationAgent:
    """Tests for the autonomous validation orchestrator."""

    def test_full_validation_passes(self):
        agent = AutoValidationAgent(seed=42)
        verdict = agent.run_full_validation(
            scalability_nodes=[100, 500, 1000],
            n_reproducibility_seeds=20,
        )
        assert verdict.overall_pass is True
        assert verdict.hardware_contract_valid is True
        assert verdict.scalability_proven is True
        assert verdict.attestation_chain_valid is True
        assert verdict.reproducibility_proven is True
        assert verdict.pass_rate == 1.0

    def test_deterministic_hash(self):
        """Same seed produces same validation hash."""
        agent1 = AutoValidationAgent(seed=42)
        v1 = agent1.run_full_validation(
            scalability_nodes=[100, 500],
            n_reproducibility_seeds=10,
        )
        agent2 = AutoValidationAgent(seed=42)
        v2 = agent2.run_full_validation(
            scalability_nodes=[100, 500],
            n_reproducibility_seeds=10,
        )
        assert v1.deterministic_hash == v2.deterministic_hash

    def test_convenience_function(self):
        verdict = run_auto_validation(
            seed=42,
            scalability_nodes=[100, 500],
            n_reproducibility_seeds=10,
        )
        assert verdict.overall_pass is True

    def test_report_structure(self):
        agent = AutoValidationAgent(seed=42)
        verdict = agent.run_full_validation(
            scalability_nodes=[100],
            n_reproducibility_seeds=5,
        )
        report = verdict.report
        assert "hardware" in report
        assert "scalability" in report
        assert "reproducibility" in report
        assert "attestation" in report
        assert "determinism" in report

    def test_verdict_to_dict(self):
        verdict = run_auto_validation(
            seed=42,
            scalability_nodes=[100],
            n_reproducibility_seeds=5,
        )
        d = verdict.to_dict()
        assert isinstance(d, dict)
        assert "overall_pass" in d
        assert "deterministic_hash" in d
        assert "execution_time_s" in d


# ===========================================================================
# Integration: Engines on the Internal Bus
# ===========================================================================


class TestEnginesOnBus:
    """Test that new engines integrate with the MESIE internal bus."""

    def test_engines_in_default_registry(self):
        from mesie.engines.registry import build_default_registry

        registry = build_default_registry()
        names = registry.names()
        assert "hardware_abstraction" in names
        assert "scalability" in names
        assert "attestation" in names
        assert "reproducibility" in names

    def test_bus_dispatch_to_scalability(self):
        from mesie.internal_api.bus import InternalBus
        from mesie.engines.registry import build_default_registry

        bus = InternalBus()
        registry = build_default_registry(bus)
        resp = bus.request("test", "scalability", "spawn_network", {"n_nodes": 50, "seed": 42})
        assert resp.ok is True
        assert resp.data["n_nodes"] == 50

    def test_bus_dispatch_to_attestation(self):
        from mesie.internal_api.bus import InternalBus
        from mesie.engines.registry import build_default_registry

        bus = InternalBus()
        registry = build_default_registry(bus)
        resp = bus.request("test", "attestation", "create_chain", {"chain_id": "bus-test"})
        assert resp.ok is True
        assert resp.data["chain_id"] == "bus-test"

    def test_bus_dispatch_to_reproducibility(self):
        from mesie.internal_api.bus import InternalBus
        from mesie.engines.registry import build_default_registry

        bus = InternalBus()
        registry = build_default_registry(bus)
        resp = bus.request("test", "reproducibility", "prove_computation", {"seed": 42})
        assert resp.ok is True
        assert resp.data["reproducible"] is True

    def test_bus_dispatch_to_hardware(self):
        from mesie.internal_api.bus import InternalBus
        from mesie.engines.registry import build_default_registry

        bus = InternalBus()
        registry = build_default_registry(bus)
        resp = bus.request("test", "hardware_abstraction", "register_device", {
            "device_class": "accelerometer",
        })
        assert resp.ok is True
        assert "device_id" in resp.data
