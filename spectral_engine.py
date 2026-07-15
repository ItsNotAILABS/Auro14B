"""Spectral matching and generation engine for Multi-Elemental records.

This module implements a production-ready foundation for MESIE architecture with:
- Ancient Engine mapping (graph and node-weight logic)
- Electro Layer feature extraction
- Multi-component spectral matching
- PSD/FAS/RotDnn compatible generation and validation
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
import pandas as pd

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - optional fallback when scipy is unavailable
    savgol_filter = None
    HAS_SCIPY = False

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None


ArrayLike = Union[np.ndarray, Sequence[float]]
RecordInput = Union["MultiElementRecord", np.ndarray, pd.DataFrame, Mapping[str, Any], str, Path]
_FALLBACK_COMPONENT_NAME = "__mesie_fallback_component__"


@dataclass
class SpectralComponent:
    """A single elemental spectral component with optional phase and node linkage."""

    name: str
    frequency: np.ndarray
    amplitude: np.ndarray
    phase: Optional[np.ndarray] = None
    domain: str = "frequency"
    units: str = "linear"
    element_weight: float = 1.0
    node_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AncientNode:
    """Ancient node metadata used for Ancient Engine mapping and lineage alignment."""

    node_id: str
    lineage_tags: List[str] = field(default_factory=list)
    symbolic_weight: float = 1.0
    resonance_group: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class ElectroSignature:
    """Computed Electro Layer feature signature for a record."""

    spectral_centroid: float
    spectral_spread: float
    band_energy: Dict[str, float]
    frequency_resonance: float
    coherence_signature: float
    harmonic_alignment: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiElementRecord:
    """Multi-elemental record containing one or more spectral components."""

    record_id: str
    components: List[SpectralComponent]
    ancient_nodes: Dict[str, AncientNode] = field(default_factory=dict)
    electro_metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    representation: str = "single"


@dataclass
class MatchResult:
    """Detailed match output between candidate and reference signal truth."""

    reference_id: str
    candidate_id: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    component_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration used by SpectralGenerator for PSD/FAS/RotDnn compatible output."""

    target_frequency: Optional[np.ndarray] = None
    amplitude_shape: str = "flat"
    stochastic_perturbation: float = 0.0
    seed: Optional[int] = None
    multi_element_blending: Dict[str, float] = field(default_factory=dict)
    ancient_node_influence: Dict[str, float] = field(default_factory=dict)
    electro_modulation: float = 0.0
    physical_min_amplitude: float = 1e-12
    physical_max_amplitude: float = 1e6
    output_format: str = "single"


@dataclass
class ValidationReport:
    """Validation report for a multi-element record with errors and warnings."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _as_float_array(values: ArrayLike, field_name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional; got shape {arr.shape}.")
    return arr


def _normalize_weights(values: Mapping[str, float], keys: Iterable[str]) -> Dict[str, float]:
    output = {k: float(values.get(k, 1.0)) for k in keys}
    s = sum(max(v, 0.0) for v in output.values())
    if s <= 0:
        return {k: 1.0 / max(len(output), 1) for k in output}
    return {k: max(v, 0.0) / s for k, v in output.items()}


def _load_from_dict(payload: Mapping[str, Any], record_id: Optional[str] = None) -> MultiElementRecord:
    rid = str(payload.get("record_id") or record_id or "record")
    representation = str(payload.get("representation", "single"))

    nodes: Dict[str, AncientNode] = {}
    for n in payload.get("ancient_nodes", []) or []:
        node = AncientNode(
            node_id=str(n["node_id"]),
            lineage_tags=list(n.get("lineage_tags", [])),
            symbolic_weight=float(n.get("symbolic_weight", 1.0)),
            resonance_group=n.get("resonance_group"),
            embedding=np.asarray(n["embedding"], dtype=float) if n.get("embedding") is not None else None,
        )
        nodes[node.node_id] = node

    components: List[SpectralComponent] = []
    if "components" in payload:
        for idx, comp in enumerate(payload["components"]):
            components.append(
                SpectralComponent(
                    name=str(comp.get("name", f"component_{idx}")),
                    frequency=_as_float_array(comp["frequency"], "frequency"),
                    amplitude=_as_float_array(comp["amplitude"], "amplitude"),
                    phase=_as_float_array(comp["phase"], "phase") if comp.get("phase") is not None else None,
                    domain=str(comp.get("domain", "frequency")),
                    units=str(comp.get("units", "linear")),
                    element_weight=float(comp.get("element_weight", 1.0)),
                    node_id=comp.get("node_id"),
                    metadata=dict(comp.get("metadata", {})),
                )
            )
    elif "frequency" in payload and "amplitude" in payload:
        components.append(
            SpectralComponent(
                name=str(payload.get("name", "component_0")),
                frequency=_as_float_array(payload["frequency"], "frequency"),
                amplitude=_as_float_array(payload["amplitude"], "amplitude"),
                phase=_as_float_array(payload["phase"], "phase") if payload.get("phase") is not None else None,
                units=str(payload.get("units", "linear")),
                node_id=payload.get("node_id"),
            )
        )
    else:
        raise ValueError("Dictionary input must provide either 'components' or 'frequency'/'amplitude'.")

    return MultiElementRecord(
        record_id=rid,
        components=components,
        ancient_nodes=nodes,
        electro_metadata=dict(payload.get("electro_metadata", {})),
        lineage=list(payload.get("lineage", [])),
        representation=representation,
    )


def load_record(source: RecordInput, record_id: Optional[str] = None) -> MultiElementRecord:
    """Load a multi-element record from arrays, dataframe, CSV, JSON, dict, or record object."""

    if isinstance(source, MultiElementRecord):
        return source

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            source = pd.read_csv(path)
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                return _load_from_dict(json.load(f), record_id=record_id)
        else:
            raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or JSON.")

    if isinstance(source, pd.DataFrame):
        lower_cols = {c.lower(): c for c in source.columns}
        if "frequency" not in lower_cols:
            raise ValueError("DataFrame input must contain a 'frequency' column.")
        fcol = lower_cols["frequency"]
        acols = [c for c in source.columns if c != fcol]
        if not acols:
            raise ValueError("DataFrame must contain at least one amplitude column.")
        components = [
            SpectralComponent(
                name=str(c),
                frequency=_as_float_array(source[fcol].to_numpy(), "frequency"),
                amplitude=_as_float_array(source[c].to_numpy(), f"amplitude[{c}]"),
            )
            for c in acols
        ]
        representation = "single" if len(components) == 1 else "multi"
        return MultiElementRecord(record_id=record_id or "record", components=components, representation=representation)

    if isinstance(source, Mapping):
        return _load_from_dict(source, record_id=record_id)

    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=float)
        if arr.ndim == 1:
            freq = np.arange(arr.shape[0], dtype=float)
            comp = SpectralComponent(name="component_0", frequency=freq, amplitude=arr)
            return MultiElementRecord(record_id=record_id or "record", components=[comp], representation="single")
        if arr.ndim == 2 and arr.shape[1] >= 2:
            comp = SpectralComponent(name="component_0", frequency=arr[:, 0], amplitude=arr[:, 1])
            return MultiElementRecord(record_id=record_id or "record", components=[comp], representation="single")
        raise ValueError("NumPy input must be 1D amplitude or 2D array with frequency and amplitude columns.")

    raise TypeError(f"Unsupported input type: {type(source)!r}")


def normalize_component(component: SpectralComponent, method: str = "max") -> SpectralComponent:
    """Normalize a spectral component amplitude with max, L2, or z-score strategy."""

    amp = component.amplitude.astype(float)
    if method == "max":
        scale = max(float(np.max(np.abs(amp))), 1e-12)
        out = amp / scale
    elif method == "l2":
        scale = max(float(np.linalg.norm(amp)), 1e-12)
        out = amp / scale
    elif method == "zscore":
        std = max(float(np.std(amp)), 1e-12)
        out = (amp - float(np.mean(amp))) / std
    else:
        raise ValueError(f"Unsupported normalization method '{method}'.")
    return SpectralComponent(**{**component.__dict__, "amplitude": out})


def smooth_component(component: SpectralComponent, window_length: int = 9, polyorder: int = 2) -> SpectralComponent:
    """Smooth component amplitude using Savitzky-Golay (or moving average fallback)."""

    amp = component.amplitude.astype(float)
    if len(amp) < 3:
        return component
    wl = max(3, (int(window_length) // 2) * 2 + 1)
    if wl > len(amp):
        wl = len(amp) if len(amp) % 2 == 1 else len(amp) - 1
    if wl < 3:
        return component

    if HAS_SCIPY and savgol_filter is not None and wl > polyorder:
        smoothed = savgol_filter(amp, window_length=wl, polyorder=min(polyorder, wl - 1))
    else:
        kernel = np.ones(wl, dtype=float) / wl
        smoothed = np.convolve(amp, kernel, mode="same")

    return SpectralComponent(**{**component.__dict__, "amplitude": smoothed})


def interpolate_component(
    component: SpectralComponent,
    target_frequency: ArrayLike,
    log_frequency: bool = False,
) -> SpectralComponent:
    """Interpolate a component onto a target frequency grid with optional log-frequency handling."""

    target = _as_float_array(target_frequency, "target_frequency")
    src_f = component.frequency
    src_a = component.amplitude

    if log_frequency:
        if np.any(src_f <= 0) or np.any(target <= 0):
            raise ValueError("Log-frequency interpolation requires strictly positive frequencies.")
        x_src = np.log10(src_f)
        x_tgt = np.log10(target)
    else:
        x_src = src_f
        x_tgt = target

    amp = np.interp(x_tgt, x_src, src_a, left=src_a[0], right=src_a[-1])
    phase = None
    if component.phase is not None:
        phase = np.interp(x_tgt, x_src, component.phase, left=component.phase[0], right=component.phase[-1])

    return SpectralComponent(**{**component.__dict__, "frequency": target, "amplitude": amp, "phase": phase})


def scale_component_amplitude(component: SpectralComponent, scale: float) -> SpectralComponent:
    """Apply amplitude scaling to a component."""

    return SpectralComponent(**{**component.__dict__, "amplitude": component.amplitude * float(scale)})


def normalize_record(record: RecordInput, method: str = "max") -> MultiElementRecord:
    """Load and normalize every component in a Multi-Elemental record."""

    rec = load_record(record)
    return MultiElementRecord(
        record_id=rec.record_id,
        components=[normalize_component(c, method=method) for c in rec.components],
        ancient_nodes=rec.ancient_nodes,
        electro_metadata=rec.electro_metadata,
        lineage=rec.lineage,
        representation=rec.representation,
    )


class AncientEngineMapper:
    """Ancient Engine layer implementing inspectable node and graph-based weighting logic."""

    def __init__(self, node_graph: Optional[Mapping[str, Sequence[str]]] = None) -> None:
        self.node_graph = {k: list(v) for k, v in (node_graph or {}).items()}
        self._graph = None
        if nx is not None and self.node_graph:
            g = nx.Graph()
            for node, neighbors in self.node_graph.items():
                g.add_node(node)
                for n in neighbors:
                    g.add_edge(node, n)
            self._graph = g

    def map_component_weight(
        self,
        record: MultiElementRecord,
        component: SpectralComponent,
        node_weight_overrides: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Compute Ancient Node Mapping weight for a component."""

        base = max(component.element_weight, 0.0)
        if not component.node_id:
            return base

        node = record.ancient_nodes.get(component.node_id)
        if node is None:
            return base

        override = float((node_weight_overrides or {}).get(node.node_id, 1.0))
        symbolic = max(node.symbolic_weight, 0.0)
        centrality = 1.0
        if self._graph is not None and node.node_id in self._graph:
            centrality += float(nx.degree_centrality(self._graph).get(node.node_id, 0.0))
        return base * symbolic * override * centrality

    def alignment_score(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> float:
        """Compute ancient-node alignment score based on weighted overlap of node identifiers."""

        ref_nodes = {c.node_id for c in reference.components if c.node_id}
        cand_nodes = {c.node_id for c in candidate.components if c.node_id}
        if not ref_nodes and not cand_nodes:
            return 1.0
        if not ref_nodes or not cand_nodes:
            return 0.0

        inter = len(ref_nodes & cand_nodes)
        union = len(ref_nodes | cand_nodes)
        if union == 0:
            return 0.0
        return inter / union


class ElectroLayer:
    """Electro Layer feature computation and electro-distance metrics."""

    def __init__(self, band_edges: Optional[Sequence[Tuple[float, float, str]]] = None) -> None:
        self.band_edges = list(
            band_edges
            or [
                (0.0, 1.0, "band_low"),
                (1.0, 10.0, "band_mid"),
                (10.0, 100.0, "band_high"),
                (100.0, np.inf, "band_ultra"),
            ]
        )

    def _aggregate(self, record: MultiElementRecord, target_frequency: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not record.components:
            return np.array([], dtype=float), np.array([], dtype=float)

        base_grid = target_frequency if target_frequency is not None else record.components[0].frequency
        amp = np.zeros_like(base_grid, dtype=float)
        for c in record.components:
            ci = interpolate_component(c, base_grid)
            amp += np.abs(ci.amplitude) * max(c.element_weight, 0.0)
        return base_grid, amp

    def compute_signature(self, record: MultiElementRecord) -> ElectroSignature:
        """Compute Electro signature including resonance, centroid, spread, and coherence."""

        freq, amp = self._aggregate(record)
        if len(freq) == 0:
            return ElectroSignature(0.0, 0.0, {}, 0.0, 0.0, 0.0)

        total = max(float(np.sum(amp)), 1e-12)
        centroid = float(np.sum(freq * amp) / total)
        spread = float(np.sqrt(np.sum(((freq - centroid) ** 2) * amp) / total))

        band_energy: Dict[str, float] = {}
        for low, high, name in self.band_edges:
            mask = (freq >= low) & (freq < high)
            band_energy[name] = float(np.sum(amp[mask]))

        resonance = float(np.max(amp) / max(float(np.mean(amp)), 1e-12))

        if len(record.components) > 1:
            stack = np.vstack([interpolate_component(c, freq).amplitude for c in record.components])
            coherence = float(1.0 / (1.0 + np.mean(np.std(stack, axis=0))))
        else:
            coherence = 1.0

        peak_idx = int(np.argmax(amp))
        f_peak = max(float(freq[peak_idx]), 1e-12)
        base = max(float(np.min(freq[freq > 0])) if np.any(freq > 0) else f_peak, 1e-12)
        harmonic_ratio = f_peak / base
        harmonic_alignment = float(1.0 / (1.0 + abs(harmonic_ratio - round(harmonic_ratio))))

        return ElectroSignature(
            spectral_centroid=centroid,
            spectral_spread=spread,
            band_energy=band_energy,
            frequency_resonance=resonance,
            coherence_signature=coherence,
            harmonic_alignment=harmonic_alignment,
        )

    def electro_distance(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> float:
        """Compute electro-distance between two records using Electro signatures."""

        rs = self.compute_signature(reference)
        cs = self.compute_signature(candidate)

        keys = sorted(set(rs.band_energy) | set(cs.band_energy))
        band_delta = sum((rs.band_energy.get(k, 0.0) - cs.band_energy.get(k, 0.0)) ** 2 for k in keys)
        v = np.array(
            [
                rs.spectral_centroid - cs.spectral_centroid,
                rs.spectral_spread - cs.spectral_spread,
                rs.frequency_resonance - cs.frequency_resonance,
                rs.coherence_signature - cs.coherence_signature,
                rs.harmonic_alignment - cs.harmonic_alignment,
                np.sqrt(band_delta),
            ],
            dtype=float,
        )
        return float(np.linalg.norm(v))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _band_weighted_error(freq: np.ndarray, err: np.ndarray, band_weights: Optional[Sequence[Tuple[float, float, float]]]) -> float:
    if not band_weights:
        return float(np.mean(err))
    weighted_sum = 0.0
    total_weight = 0.0
    for low, high, weight in band_weights:
        mask = (freq >= low) & (freq < high)
        if not np.any(mask):
            continue
        w = max(float(weight), 0.0)
        weighted_sum += float(np.mean(err[mask])) * w
        total_weight += w
    if total_weight <= 0:
        return float(np.mean(err))
    return weighted_sum / total_weight


class SpectralMatcher:
    """Spectral matching engine combining similarity, error, electro, and node alignment metrics."""

    def __init__(
        self,
        phase_aware: bool = False,
        band_weights: Optional[Sequence[Tuple[float, float, float]]] = None,
        channel_weights: Optional[Mapping[str, float]] = None,
        node_weight_overrides: Optional[Mapping[str, float]] = None,
        score_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.phase_aware = phase_aware
        self.band_weights = band_weights
        self.channel_weights = dict(channel_weights or {})
        self.node_weight_overrides = dict(node_weight_overrides or {})
        self.references: List[MultiElementRecord] = []
        self.mapper = AncientEngineMapper()
        self.electro = ElectroLayer()
        self.score_weights = {
            "cosine": 1.0,
            "rmse": 1.0,
            "log_distance": 1.0,
            "band_error": 1.0,
            "coherence": 1.0,
            "electro": 1.0,
            "node": 1.0,
            "phase": 0.5,
            **(score_weights or {}),
        }

    def fit(self, reference_records: Sequence[RecordInput]) -> "SpectralMatcher":
        """Fit matcher with reference records used for matching and ranking."""
        self.references = []
        for idx, record in enumerate(reference_records):
            try:
                self.references.append(load_record(record))
            except Exception as exc:
                raise ValueError(f"Failed to load reference record at index {idx}: {exc}") from exc
        return self

    def _common_grid(self, reference: MultiElementRecord, candidate: MultiElementRecord) -> np.ndarray:
        min_f = max(min(c.frequency[0] for c in reference.components), min(c.frequency[0] for c in candidate.components))
        max_f = min(max(c.frequency[-1] for c in reference.components), max(c.frequency[-1] for c in candidate.components))
        if max_f <= min_f:
            union = np.unique(np.concatenate([reference.components[0].frequency, candidate.components[0].frequency]))
            return union
        n = max(len(reference.components[0].frequency), len(candidate.components[0].frequency), 64)
        return np.linspace(min_f, max_f, n)

    def _component_map(self, record: MultiElementRecord) -> Dict[str, SpectralComponent]:
        return {c.name: c for c in record.components}

    def score(self, reference: RecordInput, candidate: RecordInput) -> MatchResult:
        """Compute comprehensive score between reference and candidate records."""

        ref = load_record(reference)
        cand = load_record(candidate)

        freq = self._common_grid(ref, cand)
        ref_map = self._component_map(ref)
        cand_map = self._component_map(cand)
        common = sorted(set(ref_map) & set(cand_map))
        if not common:
            ref_map[_FALLBACK_COMPONENT_NAME] = next(iter(ref_map.values()))
            cand_map[_FALLBACK_COMPONENT_NAME] = next(iter(cand_map.values()))
            common = [_FALLBACK_COMPONENT_NAME]

        ch_weights = _normalize_weights(self.channel_weights, common)

        ref_mix = np.zeros_like(freq)
        cand_mix = np.zeros_like(freq)
        component_scores: Dict[str, float] = {}
        phase_scores: List[float] = []

        for name in common:
            rc = interpolate_component(ref_map[name], freq)
            cc = interpolate_component(cand_map[name], freq)
            rw = self.mapper.map_component_weight(ref, rc, self.node_weight_overrides)
            cw = self.mapper.map_component_weight(cand, cc, self.node_weight_overrides)
            w = ch_weights[name]
            ref_mix += rc.amplitude * rw * w
            cand_mix += cc.amplitude * cw * w
            component_scores[name] = _cosine_similarity(rc.amplitude, cc.amplitude)
            if self.phase_aware and rc.phase is not None and cc.phase is not None:
                phase_scores.append(float(np.mean(np.cos(rc.phase - cc.phase))))

        eps = 1e-12
        diff = ref_mix - cand_mix
        abs_diff = np.abs(diff)

        cosine = _cosine_similarity(ref_mix, cand_mix)
        rmse = float(np.sqrt(np.mean(diff**2)))
        log_distance = float(np.mean(np.abs(np.log(np.abs(ref_mix) + eps) - np.log(np.abs(cand_mix) + eps))))
        band_error = _band_weighted_error(freq, abs_diff, self.band_weights)
        coherence = float(np.mean(list(component_scores.values()))) if component_scores else 0.0
        electro_similarity = 1.0 / (1.0 + self.electro.electro_distance(ref, cand))
        node_alignment = self.mapper.alignment_score(ref, cand)
        phase_similarity = float(np.mean(phase_scores)) if phase_scores else 1.0

        score_terms = {
            "cosine": max(cosine, 0.0),
            "rmse": 1.0 / (1.0 + rmse),
            "log_distance": 1.0 / (1.0 + log_distance),
            "band_error": 1.0 / (1.0 + band_error),
            "coherence": max(coherence, 0.0),
            "electro": electro_similarity,
            "node": node_alignment,
            "phase": max(min(phase_similarity, 1.0), 0.0),
        }

        wsum = sum(max(float(self.score_weights.get(k, 0.0)), 0.0) for k in score_terms)
        if wsum <= 0:
            final_score = 0.0
        else:
            final_score = sum(score_terms[k] * max(float(self.score_weights.get(k, 0.0)), 0.0) for k in score_terms) / wsum

        return MatchResult(
            reference_id=ref.record_id,
            candidate_id=cand.record_id,
            score=float(final_score),
            metrics={
                "cosine_similarity": float(cosine),
                "rmse": rmse,
                "log_spectral_distance": log_distance,
                "frequency_band_weighted_error": band_error,
                "elemental_channel_coherence": coherence,
                "electro_signature_similarity": electro_similarity,
                "ancient_node_alignment": node_alignment,
                "phase_similarity": phase_similarity,
            },
            component_scores=component_scores,
        )

    def match(self, candidate_record: RecordInput) -> MatchResult:
        """Match a candidate against fitted references and return the best result."""

        if not self.references:
            raise RuntimeError("SpectralMatcher.fit must be called before match.")
        candidate = load_record(candidate_record)
        results = [self.score(ref, candidate) for ref in self.references]
        return max(results, key=lambda r: r.score)

    def batch_match(self, candidate_records: Sequence[RecordInput]) -> List[MatchResult]:
        """Match a batch of candidate records against fitted references."""

        return [self.match(c) for c in candidate_records]

    def rank_matches(self, candidate: RecordInput, top_k: int = 10) -> List[MatchResult]:
        """Rank all fitted references against a candidate and return top-k results."""

        if not self.references:
            raise RuntimeError("SpectralMatcher.fit must be called before rank_matches.")
        cand = load_record(candidate)
        ranked = sorted((self.score(r, cand) for r in self.references), key=lambda x: x.score, reverse=True)
        return ranked[: max(int(top_k), 1)]


def _default_frequency_grid() -> np.ndarray:
    return np.logspace(-1, 2, 256)


class SpectralGenerator:
    """Generator for single, RotDnn, PSD, and FAS compatible multi-element spectra."""

    def __init__(self) -> None:
        self.mapper = AncientEngineMapper()

    def _shape(self, freq: np.ndarray, shape: str) -> np.ndarray:
        if shape == "flat":
            return np.ones_like(freq)
        if shape == "power_law":
            return 1.0 / np.sqrt(np.maximum(freq, 1e-6))
        if shape == "gaussian":
            center = np.exp(np.mean(np.log(np.maximum(freq, 1e-6))))
            sigma = center / 2.5
            return np.exp(-0.5 * ((freq - center) / max(sigma, 1e-6)) ** 2)
        raise ValueError(f"Unsupported amplitude_shape '{shape}'.")

    def generate(self, config: GenerationConfig, base_record: Optional[RecordInput] = None) -> MultiElementRecord:
        """Generate synthetic spectral records under MESIE architecture constraints."""

        rng = np.random.default_rng(config.seed)
        if config.target_frequency is not None:
            freq = _as_float_array(config.target_frequency, "target_frequency")
        elif base_record is not None:
            freq = load_record(base_record).components[0].frequency
        else:
            freq = _default_frequency_grid()

        base_amp = self._shape(freq, config.amplitude_shape)
        if config.stochastic_perturbation > 0:
            noise = rng.normal(0.0, float(config.stochastic_perturbation), size=len(freq))
            base_amp = base_amp * np.exp(noise)

        if config.electro_modulation != 0.0:
            modulation = 1.0 + float(config.electro_modulation) * np.sin(2 * np.pi * np.log10(np.maximum(freq, 1e-6)))
            base_amp = base_amp * np.maximum(modulation, 1e-6)

        blend = config.multi_element_blending or {"component_0": 1.0}
        blend = _normalize_weights(blend, blend.keys())
        output_units = {"psd": "psd", "fas": "fas"}

        components: List[SpectralComponent] = []
        for name, w in blend.items():
            amp = base_amp * w
            node_weight = float(config.ancient_node_influence.get(name, 1.0))
            amp = amp * max(node_weight, 0.0)
            amp = np.clip(amp, config.physical_min_amplitude, config.physical_max_amplitude)
            units = output_units.get(config.output_format.lower(), "linear")
            components.append(
                SpectralComponent(
                    name=name,
                    frequency=freq,
                    amplitude=amp,
                    units=units,
                    node_id=name if name in config.ancient_node_influence else None,
                )
            )

        representation = config.output_format.lower()
        if representation not in {"single", "rotdnn", "psd", "fas", "multi"}:
            representation = "single"

        return MultiElementRecord(
            record_id="generated_record",
            components=components,
            representation=representation,
            lineage=["generated", "synthetic"],
        )


class _ValidationRules:
    PSD_UNITS = {"psd", "power/hz", "(m/s^2)^2/hz", "g^2/hz"}
    FAS_UNITS = {"fas", "m/s", "cm/s", "unit/sqrt(hz)"}


def validate_record(record: RecordInput) -> ValidationReport:
    """Validate record for frequency integrity, dimensions, finite values, and format consistency."""

    rec = load_record(record)
    errors: List[str] = []
    warnings: List[str] = []

    if not rec.components:
        errors.append("Record contains no spectral components.")

    base_len = None
    base_freq = None
    for idx, c in enumerate(rec.components):
        if len(c.frequency) != len(c.amplitude):
            errors.append(f"Component '{c.name}' has mismatched frequency and amplitude array lengths.")
            continue

        if len(c.frequency) == 0:
            errors.append(f"Component '{c.name}' has empty frequency data.")
            continue

        if np.any(~np.isfinite(c.frequency)) or np.any(~np.isfinite(c.amplitude)):
            errors.append(f"Component '{c.name}' contains NaN/Inf values.")

        if np.any(np.diff(c.frequency) <= 0):
            errors.append(f"Component '{c.name}' has non-monotonically increasing frequency values.")

        if np.any(c.amplitude < 0):
            errors.append(f"Component '{c.name}' contains negative amplitudes.")

        if np.any(np.diff(c.frequency) > np.median(np.diff(c.frequency)) * 5):
            warnings.append(f"Component '{c.name}' may have missing frequencies due to large grid gaps.")

        if base_len is None:
            base_len = len(c.frequency)
            base_freq = c.frequency
        else:
            if len(c.frequency) != base_len:
                warnings.append("Incompatible component dimensions across components.")
            elif not np.allclose(c.frequency, base_freq):
                warnings.append("Components are on different frequency grids.")

    rep = rec.representation.lower()
    if rep == "psd":
        for c in rec.components:
            if c.units.lower() not in _ValidationRules.PSD_UNITS:
                warnings.append(f"PSD unit compatibility warning: '{c.name}' uses '{c.units}' for PSD representation.")
    if rep == "fas":
        for c in rec.components:
            if c.units.lower() not in _ValidationRules.FAS_UNITS:
                warnings.append(f"FAS unit compatibility warning: '{c.name}' uses '{c.units}' for FAS representation.")

    if rep == "rotdnn":
        names = {c.name.lower() for c in rec.components}
        if not any(("rot" in n or "rotd" in n) for n in names):
            warnings.append("RotDnn component consistency checks: expected RotDnn-like component names.")
        if len(rec.components) < 2:
            warnings.append("RotDnn component consistency checks: expected multiple rotational components.")

    return ValidationReport(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def match_records(
    reference: RecordInput,
    candidate: RecordInput,
    matcher: Optional[SpectralMatcher] = None,
) -> MatchResult:
    """Convenience API: match two records and return a comprehensive MatchResult."""

    m = matcher or SpectralMatcher()
    return m.score(reference, candidate)


def generate_psd(config: GenerationConfig, base_record: Optional[RecordInput] = None) -> MultiElementRecord:
    """Generate a PSD-compatible spectral record."""

    cfg = replace(config, output_format="psd")
    return SpectralGenerator().generate(cfg, base_record=base_record)


def generate_fas(config: GenerationConfig, base_record: Optional[RecordInput] = None) -> MultiElementRecord:
    """Generate an FAS-compatible spectral record."""

    cfg = replace(config, output_format="fas")
    return SpectralGenerator().generate(cfg, base_record=base_record)


def generate_rotdnn(config: GenerationConfig, base_record: Optional[RecordInput] = None) -> MultiElementRecord:
    """Generate a RotDnn-compatible spectral record."""

    cfg = replace(config, output_format="rotdnn")
    if not cfg.multi_element_blending:
        cfg.multi_element_blending = {"RotD50": 0.5, "RotD100": 0.5}
    return SpectralGenerator().generate(cfg, base_record=base_record)


__all__ = [
    "AncientEngineMapper",
    "AncientNode",
    "ElectroLayer",
    "ElectroSignature",
    "GenerationConfig",
    "MatchResult",
    "MultiElementRecord",
    "SpectralComponent",
    "SpectralGenerator",
    "SpectralMatcher",
    "ValidationReport",
    "generate_fas",
    "generate_psd",
    "generate_rotdnn",
    "load_record",
    "match_records",
    "normalize_record",
    "validate_record",
]


if __name__ == "__main__":
    # Example usage for unit-test-ready demonstration of MESIE APIs.
    freq = np.logspace(-1, 2, 128)
    ref_payload = {
        "record_id": "reference_signal",
        "representation": "multi",
        "components": [
            {
                "name": "element_alpha",
                "frequency": freq.tolist(),
                "amplitude": (1.0 / np.sqrt(freq)).tolist(),
                "node_id": "node_alpha",
            },
            {
                "name": "element_beta",
                "frequency": freq.tolist(),
                "amplitude": (np.exp(-((freq - 10.0) ** 2) / 40.0)).tolist(),
                "node_id": "node_beta",
            },
        ],
        "ancient_nodes": [
            {"node_id": "node_alpha", "symbolic_weight": 1.1, "lineage_tags": ["L1"]},
            {"node_id": "node_beta", "symbolic_weight": 0.9, "lineage_tags": ["L2"]},
        ],
        "lineage": ["Ancient Engine", "Electro Layer"],
    }

    candidate_payload = {
        "record_id": "candidate_signal",
        "representation": "multi",
        "components": [
            {
                "name": "element_alpha",
                "frequency": freq.tolist(),
                "amplitude": (1.02 / np.sqrt(freq)).tolist(),
                "node_id": "node_alpha",
            },
            {
                "name": "element_beta",
                "frequency": freq.tolist(),
                "amplitude": (1.01 * np.exp(-((freq - 10.0) ** 2) / 40.0)).tolist(),
                "node_id": "node_beta",
            },
        ],
    }

    reference = load_record(ref_payload)
    candidate = normalize_record(candidate_payload)

    validation = validate_record(reference)
    print("Validation:", validation)

    matcher = SpectralMatcher(phase_aware=False)
    result = matcher.score(reference, candidate)
    print("Match score:", result.score)
    print("Match metrics:", result.metrics)

    gen_cfg = GenerationConfig(amplitude_shape="power_law", output_format="psd", seed=7)
    generated_psd = generate_psd(gen_cfg)
    print("Generated PSD record:", generated_psd.record_id, generated_psd.representation)
