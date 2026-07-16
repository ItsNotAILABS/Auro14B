from __future__ import annotations

import torch

from mesie.compute import MESIESpectralConditioner, MESIESpectralProjector, MESIESpectralRegularizer


def test_projector_shapes_and_gradients() -> None:
    hidden = torch.randn(2, 64, 32, requires_grad=True)
    projector = MESIESpectralProjector(bands=8)
    state = projector(hidden)
    assert state.band_energy.shape == (2, 8, 32)
    assert state.pooled_features.shape == (2, 13)
    state.pooled_features.mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_regularizer_is_differentiable() -> None:
    previous = torch.randn(1, 48, 24)
    hidden = torch.randn(1, 48, 24, requires_grad=True)
    loss, metrics = MESIESpectralRegularizer(bands=6)(hidden, previous)
    loss.backward()
    assert loss.ndim == 0
    assert "mesie_entropy" in metrics
    assert hidden.grad is not None


def test_conditioner_preserves_shape() -> None:
    hidden = torch.randn(2, 32, 16)
    output, metadata = MESIESpectralConditioner(hidden_size=16, bands=4)(hidden)
    assert output.shape == hidden.shape
    assert 0.0 < metadata["gate"] < 1.0
