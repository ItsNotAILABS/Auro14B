from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from auro_native_llm.model.prewiring import PrewiringConfig, apply_structured_prewiring


class FakeTokenizer:
    def encode(self, text, **_kwargs):
        return [sum(text.encode("utf-8")) % 64]


class FakeEmbedding:
    def __init__(self, rows=64, cols=16):
        self.token_embeddings = np.zeros((rows, cols), dtype=np.float32)


class FakeCore:
    def __init__(self):
        self.embedding = FakeEmbedding()
        self.tie_embeddings = True


@dataclass
class FakeConfig:
    seed: int = 42
    parameter_target: int = 4_000_000_000


class FakeModel:
    model_id = "Auro-4B"
    num_params = 1024

    def __init__(self):
        self.config = FakeConfig()
        self.tokenizer = FakeTokenizer()
        self.core = FakeCore()


def test_prewiring_is_deterministic():
    left = FakeModel()
    right = FakeModel()
    config = PrewiringConfig(seed=873539)
    left_receipt = apply_structured_prewiring(left, config)
    right_receipt = apply_structured_prewiring(right, config)
    np.testing.assert_array_equal(left.core.embedding.token_embeddings, right.core.embedding.token_embeddings)
    assert left_receipt.manifest_sha256 == right_receipt.manifest_sha256
    assert left_receipt.transform_diagnostics == right_receipt.transform_diagnostics


def test_prewiring_changes_baseline_and_preserves_shape():
    model = FakeModel()
    before = model.core.embedding.token_embeddings.copy()
    receipt = apply_structured_prewiring(model, PrewiringConfig(seed=9))
    assert model.core.embedding.token_embeddings.shape == before.shape
    assert not np.array_equal(model.core.embedding.token_embeddings, before)
    assert float(np.linalg.norm(model.core.embedding.token_embeddings)) > 0.0
    assert receipt.tensor_shapes["token_embeddings"] == [64, 16]


def test_manifest_records_native_walsh_invariants_and_claim_boundary():
    model = FakeModel()
    receipt = apply_structured_prewiring(model)
    manifest = model.prewiring_manifest
    transform = manifest["orthogonal_transform"]
    assert manifest["model_id"] == "Auro-4B"
    assert manifest["external_model_fallback"] is False
    assert manifest["claims"]["factual_knowledge_without_training"] is False
    assert transform["name"] == "normalized_walsh_hadamard"
    assert transform["ordering"] == "sequency"
    assert transform["diagnostics"]["orthogonality_max_error"] < 1e-12
    assert transform["diagnostics"]["involution_max_error"] < 1e-12
    assert transform["diagnostics"]["energy_error"] < 1e-12
    assert "normalized_walsh_hadamard_embedding_basis" in receipt.applied_components


def test_control_families_touch_multiple_embedding_rows():
    model = FakeModel()
    receipt = apply_structured_prewiring(model, PrewiringConfig(seed=27))
    touched = np.count_nonzero(np.linalg.norm(model.core.embedding.token_embeddings, axis=1))
    assert touched > 8
    assert any(note.startswith("control_token_rows_touched=") for note in receipt.notes)
