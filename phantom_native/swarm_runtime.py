"""SovereignSwarmRuntime — Native runtime for MESIE neuronet swarms.

Orchestrates multiple SovereignNeuroCores with:
- QSHA-based identity and commitment
- Shadow Wire topology masking
- Sealed intent execution
- Swarm-level consensus and aggregation

Zero external dependencies in the core path.
"""

from __future__ import annotations

import hashlib
import struct
import time
from typing import Any, Dict, List, Optional

from phantom_native.neurocore import SovereignNeuroCore
from phantom_native.sovereign_tensor import SovereignTensor


class ExecutionReceipt:
    """Public proof of swarm execution without revealing internal topology.

    Attributes:
        commitment: QSHA commitment hash of aggregated outputs.
        shadow_wire: Masked topology representation.
        public_meta: Non-sensitive metadata about the execution.
        timestamp: Time of execution.
    """

    def __init__(
        self,
        commitment: str,
        shadow_wire: Dict[str, Any],
        public_meta: Optional[Dict[str, Any]] = None,
    ):
        self.commitment = commitment
        self.shadow_wire = shadow_wire
        self.public_meta = public_meta or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt for transmission."""
        return {
            "commitment": self.commitment,
            "shadow_wire": self.shadow_wire,
            "public_meta": self.public_meta,
            "timestamp": self.timestamp,
        }

    def verify(self, expected_commitment: str) -> bool:
        """Verify receipt commitment matches expected."""
        return self.commitment == expected_commitment


class ShadowWireEnvelope:
    """Topology masking for swarm privacy.

    Hides the internal structure of the neuronet swarm while providing
    verifiable execution proofs.
    """

    def mask_topology(self, core_ids: List[str]) -> Dict[str, Any]:
        """Create a masked representation of the swarm topology.

        Args:
            core_ids: List of NeuroCore identifiers.

        Returns:
            Masked topology with only aggregate properties visible.
        """
        # Hash individual IDs so topology is hidden
        masked_ids = [self._mask_id(cid) for cid in core_ids]
        # Aggregate hash
        combined = hashlib.sha256(
            "".join(masked_ids).encode()
        ).hexdigest()[:32]

        return {
            "swarm_hash": combined,
            "n_cores": len(core_ids),
            "masked": True,
            "timestamp": time.time(),
        }

    @staticmethod
    def _mask_id(core_id: str) -> str:
        """One-way mask of a core identifier."""
        return hashlib.sha256(core_id.encode()).hexdigest()[:16]


class SovereignVault:
    """Sealed intent storage and retrieval.

    Provides deterministic sealing/opening of execution intents
    for sovereign computation without external key management.
    """

    def __init__(self):
        self._sealed_store: Dict[str, bytes] = {}

    def seal_intent(self, intent: Dict[str, Any]) -> bytes:
        """Seal an intent dictionary into deterministic bytes.

        Args:
            intent: Dictionary describing the execution intent.

        Returns:
            Sealed bytes representation.
        """
        # Deterministic serialization (sorted keys)
        parts = []
        for key in sorted(intent.keys()):
            val = intent[key]
            if isinstance(val, list):
                parts.append(f"{key}:{','.join(str(v) for v in val)}")
            else:
                parts.append(f"{key}:{val}")
        payload = "|".join(parts).encode("utf-8")
        # Simple XOR-based sealing (placeholder for real crypto)
        sealed = bytes(b ^ 0x42 for b in payload)
        seal_id = hashlib.sha256(sealed).hexdigest()[:16]
        self._sealed_store[seal_id] = sealed
        return sealed

    def open_sealed_intent(self, sealed: bytes) -> Dict[str, Any]:
        """Open a sealed intent back to dictionary form.

        Args:
            sealed: Sealed bytes from seal_intent.

        Returns:
            Reconstructed intent dictionary.
        """
        payload = bytes(b ^ 0x42 for b in sealed)
        decoded = payload.decode("utf-8")
        intent: Dict[str, Any] = {}
        for part in decoded.split("|"):
            if ":" in part:
                key, val = part.split(":", 1)
                # Try to parse lists
                if "," in val:
                    try:
                        intent[key] = [float(v) for v in val.split(",")]
                    except ValueError:
                        intent[key] = val
                else:
                    try:
                        intent[key] = float(val)
                    except ValueError:
                        intent[key] = val
        return intent


class SovereignSwarmRuntime:
    """Native runtime for MESIE neuronet swarms.

    Manages a fleet of SovereignNeuroCores, provides sealed intent
    execution, QSHA-protected commitments, and Shadow Wire topology
    masking for sovereign edge + swarm deployment.

    Attributes:
        vault: Sealed intent storage.
        wire: Shadow wire envelope for topology masking.
        cores: Dictionary of spawned NeuroCores keyed by QSHA ID.
    """

    def __init__(self):
        self.vault = SovereignVault()
        self.wire = ShadowWireEnvelope()
        self.cores: Dict[str, SovereignNeuroCore] = {}
        self.manifest_commitment = ""
        self._execution_log: List[ExecutionReceipt] = []

    def spawn_neuronet(self, spectral_config: Optional[Dict[str, Any]] = None) -> str:
        """Spawn a new NeuroCore in the swarm.

        Args:
            spectral_config: Configuration dictionary for the NeuroCore.

        Returns:
            QSHA identifier for the spawned core.
        """
        spectral_config = spectral_config or {}
        core = SovereignNeuroCore(spectral_config)
        core_id = self._qsha(repr(spectral_config) + str(time.time()))
        self.cores[core_id] = core
        self._update_manifest()
        return core_id

    def execute(self, tensor: SovereignTensor) -> List[SovereignTensor]:
        """Execute a tensor through all cores in the swarm.

        Args:
            tensor: Input SovereignTensor.

        Returns:
            List of output tensors (one per core).
        """
        results: List[SovereignTensor] = []
        for core in self.cores.values():
            out = core.forward(tensor)
            results.append(out)
        return results

    def execute_sealed_intent(self, sealed_intent: bytes) -> ExecutionReceipt:
        """Execute a sealed intent through the swarm with full provenance.

        Args:
            sealed_intent: Sealed bytes from SovereignVault.

        Returns:
            ExecutionReceipt with commitment proof and masked topology.
        """
        intent = self.vault.open_sealed_intent(sealed_intent)

        # Build tensor from intent's spectrum data
        spectrum = intent.get("spectrum", intent)
        if isinstance(spectrum, dict):
            tensor = SovereignTensor.from_mesie_component(spectrum)
        else:
            # Fallback: use all numeric values as amplitude
            amp = [v for v in intent.values() if isinstance(v, (int, float))]
            if not amp:
                amp = [0.0]
            tensor = SovereignTensor(amp, (len(amp),))

        # Execute through all cores
        results = self.execute(tensor)

        # Compute public commitment
        commitment = self._compute_commitment(results)
        shadow = self.wire.mask_topology(list(self.cores.keys()))

        receipt = ExecutionReceipt(
            commitment=commitment,
            shadow_wire=shadow,
            public_meta={
                "swarm_size": len(self.cores),
                "input_shape": tensor.shape,
                "resonance": tensor.resonance,
            },
        )
        self._execution_log.append(receipt)
        return receipt

    def aggregate_swarm(self, results: List[SovereignTensor]) -> SovereignTensor:
        """Aggregate outputs from multiple cores into a single tensor.

        Uses resonance-weighted averaging across core outputs.
        """
        if not results:
            return SovereignTensor([0.0], (1,))

        # Find common length (use minimum)
        min_len = min(len(r.data) for r in results)
        aggregated = [0.0] * min_len
        total_resonance = 0.0

        for r in results:
            weight = r.resonance
            total_resonance += weight
            for i in range(min_len):
                aggregated[i] += r.data[i] * weight

        if total_resonance > 0:
            aggregated = [x / total_resonance for x in aggregated]

        return SovereignTensor(aggregated, (min_len,), {"resonance": 1.0})

    def get_swarm_status(self) -> Dict[str, Any]:
        """Return current swarm status and metrics."""
        return {
            "n_cores": len(self.cores),
            "manifest": self.manifest_commitment,
            "executions": len(self._execution_log),
            "core_ids": list(self.cores.keys()),
        }

    def _qsha(self, data: str) -> str:
        """Compute QSHA identifier (SHA-256 based placeholder)."""
        return "qsha:" + hashlib.sha256(data.encode()).hexdigest()[:32]

    def _compute_commitment(self, results: List[SovereignTensor]) -> str:
        """Compute aggregate QSHA commitment from execution results."""
        hasher = hashlib.sha256()
        for r in results:
            hasher.update(r.to_bytes())
        return "commit:" + hasher.hexdigest()[:32]

    def _update_manifest(self) -> None:
        """Update manifest commitment when swarm topology changes."""
        hasher = hashlib.sha256()
        for core_id in sorted(self.cores.keys()):
            hasher.update(core_id.encode())
        self.manifest_commitment = "manifest:" + hasher.hexdigest()[:32]

    def __repr__(self) -> str:
        return (
            f"SovereignSwarmRuntime(cores={len(self.cores)}, "
            f"manifest={self.manifest_commitment[:24]}...)"
        )
