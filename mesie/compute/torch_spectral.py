from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class MESIESpectralState:
    """Differentiable spectral state extracted from a hidden-state sequence."""

    band_energy: Tensor
    entropy: Tensor
    centroid: Tensor
    flatness: Tensor
    high_frequency_ratio: Tensor
    phase_coherence: Tensor
    pooled_features: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "spectral_entropy": float(self.entropy.detach().mean().cpu()),
            "spectral_centroid": float(self.centroid.detach().mean().cpu()),
            "spectral_flatness": float(self.flatness.detach().mean().cpu()),
            "high_frequency_ratio": float(self.high_frequency_ratio.detach().mean().cpu()),
            "phase_coherence": float(self.phase_coherence.detach().mean().cpu()),
        }


class MESIESpectralProjector(nn.Module):
    """Project language-model hidden states into MESIE frequency-domain state.

    The sequence dimension is treated as the sampled signal axis. The operation is
    fully differentiable and can therefore be used inside a transformer training
    objective rather than only as an offline NumPy diagnostic.
    """

    def __init__(self, bands: int = 16, eps: float = 1e-8) -> None:
        super().__init__()
        if bands < 2:
            raise ValueError("bands must be at least 2")
        self.bands = bands
        self.eps = eps

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> MESIESpectralState:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch, sequence, hidden]")
        batch, sequence, _ = hidden.shape
        if sequence < 4:
            raise ValueError("sequence length must be at least 4")

        signal = hidden.float()
        if attention_mask is not None:
            if attention_mask.shape[:2] != hidden.shape[:2]:
                raise ValueError("attention_mask must match [batch, sequence]")
            mask = attention_mask.to(signal.dtype).unsqueeze(-1)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (signal * mask).sum(dim=1, keepdim=True) / count
            signal = (signal - mean) * mask
        else:
            signal = signal - signal.mean(dim=1, keepdim=True)

        window = torch.hann_window(sequence, device=signal.device, dtype=signal.dtype)
        signal = signal * window.view(1, sequence, 1)
        spectrum = torch.fft.rfft(signal, dim=1, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        frequencies = torch.linspace(0.0, 1.0, power.shape[1], device=power.device, dtype=power.dtype)

        total_power = power.sum(dim=1).clamp_min(self.eps)
        distribution = power / total_power.unsqueeze(1)
        entropy = -(distribution * (distribution + self.eps).log()).sum(dim=1)
        entropy = entropy / torch.log(torch.tensor(float(power.shape[1]), device=power.device, dtype=power.dtype))
        centroid = (distribution * frequencies.view(1, -1, 1)).sum(dim=1)
        flatness = torch.exp(torch.log(power + self.eps).mean(dim=1)) / (power.mean(dim=1) + self.eps)
        high = power[:, frequencies >= 0.25].sum(dim=1) / total_power

        unit_phase = spectrum / (spectrum.abs() + self.eps)
        phase_coherence = unit_phase.mean(dim=1).abs()

        edges = torch.linspace(0.0, 1.0, self.bands + 1, device=power.device, dtype=power.dtype).square()
        band_values: list[Tensor] = []
        for index in range(self.bands):
            left = edges[index]
            right = edges[index + 1]
            selector = (frequencies >= left) & (
                (frequencies < right) if index < self.bands - 1 else (frequencies <= right)
            )
            if not bool(selector.any()):
                nearest = torch.argmin((frequencies - ((left + right) / 2)).abs())
                selector = torch.zeros_like(frequencies, dtype=torch.bool)
                selector[nearest] = True
            band_values.append(power[:, selector].mean(dim=1))
        band_energy = torch.stack(band_values, dim=1)
        band_energy = band_energy / band_energy.sum(dim=1, keepdim=True).clamp_min(self.eps)

        pooled_features = torch.cat(
            [
                band_energy.mean(dim=-1),
                entropy.mean(dim=-1, keepdim=True),
                centroid.mean(dim=-1, keepdim=True),
                flatness.mean(dim=-1, keepdim=True),
                high.mean(dim=-1, keepdim=True),
                phase_coherence.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        if pooled_features.shape != (batch, self.bands + 5):
            raise RuntimeError(f"unexpected pooled feature shape: {pooled_features.shape}")

        return MESIESpectralState(
            band_energy=band_energy,
            entropy=entropy,
            centroid=centroid,
            flatness=flatness,
            high_frequency_ratio=high,
            phase_coherence=phase_coherence,
            pooled_features=pooled_features,
        )


class MESIESpectralRegularizer(nn.Module):
    """Auxiliary MESIE objective for transformer hidden states.

    It penalizes spectral collapse, excessive high-frequency aliasing, and abrupt
    cross-layer spectral drift. Targets are explicit and receipt-friendly rather
    than hidden heuristics.
    """

    def __init__(
        self,
        bands: int = 16,
        min_entropy: float = 0.35,
        max_high_frequency_ratio: float = 0.60,
        collapse_weight: float = 1.0,
        alias_weight: float = 0.5,
        cross_layer_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.projector = MESIESpectralProjector(bands=bands)
        self.min_entropy = min_entropy
        self.max_high_frequency_ratio = max_high_frequency_ratio
        self.collapse_weight = collapse_weight
        self.alias_weight = alias_weight
        self.cross_layer_weight = cross_layer_weight

    def forward(
        self,
        hidden: Tensor,
        previous_hidden: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        current = self.projector(hidden, attention_mask)
        collapse = F.relu(self.min_entropy - current.entropy).mean()
        alias = F.relu(current.high_frequency_ratio - self.max_high_frequency_ratio).mean()
        drift = hidden.new_zeros((), dtype=torch.float32)
        if previous_hidden is not None:
            previous = self.projector(previous_hidden, attention_mask)
            drift = F.smooth_l1_loss(current.band_energy, previous.band_energy.detach())
        total = (
            self.collapse_weight * collapse
            + self.alias_weight * alias
            + self.cross_layer_weight * drift
        )
        metrics = {
            "mesie_loss": total,
            "mesie_collapse": collapse,
            "mesie_alias": alias,
            "mesie_cross_layer_drift": drift,
            "mesie_entropy": current.entropy.mean(),
            "mesie_centroid": current.centroid.mean(),
            "mesie_phase_coherence": current.phase_coherence.mean(),
        }
        return total, metrics


class MESIESpectralConditioner(nn.Module):
    """Inject a gated spectral summary back into transformer hidden states."""

    def __init__(self, hidden_size: int, bands: int = 16, gate_init: float = -4.0) -> None:
        super().__init__()
        self.projector = MESIESpectralProjector(bands=bands)
        feature_size = bands + 5
        self.projection = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, dict[str, Any]]:
        state = self.projector(hidden, attention_mask)
        conditioning = self.projection(state.pooled_features).unsqueeze(1)
        gate = torch.sigmoid(self.gate_logit)
        output = hidden + gate.to(hidden.dtype) * conditioning.to(hidden.dtype)
        metadata = {"gate": float(gate.detach().cpu()), **state.detached_metrics()}
        return output, metadata
