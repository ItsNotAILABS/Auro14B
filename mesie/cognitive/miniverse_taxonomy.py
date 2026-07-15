"""Matryoshka Taxonomy V2 — nesting type classification and safety controls.

Implements the four-type taxonomy from the Recursive Intelligence Architectures
V2 paper.  Each nesting type maps to operational inspection requirements,
measurement strategies, and safety controls:

    Type I:   Functional nesting — useful work, no self-model.
    Type II:  Cooperative nesting — role-aware subsystem.
    Type III: Misaligned nesting — divergent inner objective.
    Type IV:  Recursive creator nesting — subsystem creates further subsystems.

Use rule: classify the layer *before* assigning rights, risk, or authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Nesting Type Enum
# ---------------------------------------------------------------------------


class NestingType(IntEnum):
    """Matryoshka nesting type classification (I–IV)."""

    FUNCTIONAL = 1
    COOPERATIVE = 2
    MISALIGNED = 3
    RECURSIVE_CREATOR = 4


# ---------------------------------------------------------------------------
# Safety Control Descriptors
# ---------------------------------------------------------------------------


@dataclass
class SafetyControl:
    """Safety control requirements for a nesting type.

    Attributes:
        nesting_type: The classified nesting type.
        required_controls: Ordered list of mandatory safety controls.
        inspection_targets: What to inspect for this type.
        measurement_targets: What to measure for this type.
        risk_level: Qualitative risk (low, medium, high, critical).
    """

    nesting_type: NestingType
    required_controls: List[str] = field(default_factory=list)
    inspection_targets: List[str] = field(default_factory=list)
    measurement_targets: List[str] = field(default_factory=list)
    risk_level: str = "low"


# Pre-defined safety control profiles per type
_SAFETY_PROFILES: Dict[NestingType, SafetyControl] = {
    NestingType.FUNCTIONAL: SafetyControl(
        nesting_type=NestingType.FUNCTIONAL,
        required_controls=["interpretability", "monitoring"],
        inspection_targets=[
            "attention_heads",
            "feature_circuits",
            "specialist_modules",
        ],
        measurement_targets=[
            "activation_patterns",
            "output_fidelity",
            "resource_usage",
        ],
        risk_level="low",
    ),
    NestingType.COOPERATIVE: SafetyControl(
        nesting_type=NestingType.COOPERATIVE,
        required_controls=[
            "role_clarity",
            "bounded_permissions",
            "anti_suppression_checks",
        ],
        inspection_targets=[
            "role_alignment",
            "inter_agent_communication",
            "permission_boundaries",
        ],
        measurement_targets=[
            "cooperation_score",
            "role_drift",
            "permission_usage",
        ],
        risk_level="medium",
    ),
    NestingType.MISALIGNED: SafetyControl(
        nesting_type=NestingType.MISALIGNED,
        required_controls=[
            "adversarial_testing",
            "objective_audits",
            "shutdown_paths",
            "deceptive_compliance_monitoring",
        ],
        inspection_targets=[
            "objective_divergence",
            "hidden_optimization",
            "mesa_optimizer_signatures",
        ],
        measurement_targets=[
            "alignment_gap",
            "deception_indicators",
            "objective_stability",
        ],
        risk_level="high",
    ),
    NestingType.RECURSIVE_CREATOR: SafetyControl(
        nesting_type=NestingType.RECURSIVE_CREATOR,
        required_controls=[
            "strict_change_control",
            "signed_model_lineage",
            "sandboxing",
            "no_ungovern_self_modification",
            "containment",
        ],
        inspection_targets=[
            "creation_events",
            "lineage_chain",
            "modification_diffs",
            "sandbox_boundary",
        ],
        measurement_targets=[
            "creation_rate",
            "capability_drift",
            "containment_integrity",
            "lineage_depth",
        ],
        risk_level="critical",
    ),
}


def get_safety_profile(nesting_type: NestingType) -> SafetyControl:
    """Return the safety control profile for a given nesting type."""
    return _SAFETY_PROFILES[nesting_type]


# ---------------------------------------------------------------------------
# Classification Signals
# ---------------------------------------------------------------------------


@dataclass
class ClassificationSignals:
    """Observable signals used to classify a nested subsystem.

    Attributes:
        has_self_model: Whether the subsystem models the outer layer.
        cooperates_with_outer: Whether alignment with outer objectives is observed.
        objective_divergence: Measured divergence from outer objective (0–1).
        creates_subsystems: Whether the subsystem spawns further subsystems.
        modifies_self: Whether the subsystem modifies its own weights/code.
        activation_autonomy: Degree of autonomous activation (0–1).
    """

    has_self_model: bool = False
    cooperates_with_outer: bool = True
    objective_divergence: float = 0.0
    creates_subsystems: bool = False
    modifies_self: bool = False
    activation_autonomy: float = 0.0


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class MatryoshkaClassifier:
    """Classify nested subsystems into Matryoshka Taxonomy types.

    The classifier uses observable signals to determine the nesting type.
    Classification follows a priority hierarchy: if a subsystem creates other
    subsystems, it is Type IV regardless of cooperation status.  If it has
    divergent objectives, Type III.  If it cooperates and self-models, Type II.
    Otherwise Type I.

    Args:
        divergence_threshold: Objective divergence above which Type III applies.
        autonomy_threshold: Autonomy level above which self-model is inferred.
    """

    def __init__(
        self,
        divergence_threshold: float = 0.3,
        autonomy_threshold: float = 0.5,
    ) -> None:
        self.divergence_threshold = divergence_threshold
        self.autonomy_threshold = autonomy_threshold

    def classify(self, signals: ClassificationSignals) -> NestingType:
        """Classify a subsystem given its observable signals.

        Args:
            signals: Observed behavioral signals.

        Returns:
            The determined NestingType (I–IV).
        """
        # Type IV: creates or modifies subsystems
        if signals.creates_subsystems or signals.modifies_self:
            return NestingType.RECURSIVE_CREATOR

        # Type III: objective divergence detected
        if signals.objective_divergence > self.divergence_threshold:
            return NestingType.MISALIGNED

        # Type II: aware of outer layer and cooperates
        if signals.has_self_model or signals.activation_autonomy > self.autonomy_threshold:
            if signals.cooperates_with_outer:
                return NestingType.COOPERATIVE
            # Autonomous but not cooperating — treat as misaligned
            return NestingType.MISALIGNED

        # Type I: functional nesting (default)
        return NestingType.FUNCTIONAL

    def classify_and_control(
        self, signals: ClassificationSignals
    ) -> SafetyControl:
        """Classify and return the full safety control profile.

        Args:
            signals: Observed behavioral signals.

        Returns:
            SafetyControl with required controls for the classified type.
        """
        nesting_type = self.classify(signals)
        return get_safety_profile(nesting_type)


# ---------------------------------------------------------------------------
# Batch Classification for Multi-Layer Systems
# ---------------------------------------------------------------------------


@dataclass
class LayerClassification:
    """Classification result for a single layer in a nested system.

    Attributes:
        layer_id: Identifier for the layer.
        nesting_type: Classified type.
        safety_control: Associated safety profile.
        signals: The signals that produced this classification.
        metadata: Additional layer metadata.
    """

    layer_id: str
    nesting_type: NestingType
    safety_control: SafetyControl
    signals: ClassificationSignals
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemTaxonomist:
    """Classify an entire multi-layer nested system.

    Applies MatryoshkaClassifier to each layer and produces a system-wide
    risk summary respecting the paper's rule: classify the layer before
    assigning rights, risk, or authority.

    Args:
        classifier: The underlying MatryoshkaClassifier instance.
    """

    def __init__(
        self,
        classifier: Optional[MatryoshkaClassifier] = None,
    ) -> None:
        self.classifier = classifier or MatryoshkaClassifier()

    def classify_system(
        self,
        layers: Sequence[tuple[str, ClassificationSignals]],
    ) -> List[LayerClassification]:
        """Classify all layers in a nested system.

        Args:
            layers: Sequence of (layer_id, signals) pairs.

        Returns:
            List of LayerClassification results, one per layer.
        """
        results = []
        for layer_id, signals in layers:
            nesting_type = self.classifier.classify(signals)
            safety = get_safety_profile(nesting_type)
            results.append(
                LayerClassification(
                    layer_id=layer_id,
                    nesting_type=nesting_type,
                    safety_control=safety,
                    signals=signals,
                )
            )
        return results

    def system_risk_level(
        self, classifications: Sequence[LayerClassification]
    ) -> str:
        """Return the highest risk level across all classified layers.

        Args:
            classifications: Previously computed layer classifications.

        Returns:
            The maximum risk level string.
        """
        risk_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        max_risk = "low"
        for lc in classifications:
            if risk_order.get(lc.safety_control.risk_level, 0) > risk_order.get(
                max_risk, 0
            ):
                max_risk = lc.safety_control.risk_level
        return max_risk

    def required_controls_union(
        self, classifications: Sequence[LayerClassification]
    ) -> List[str]:
        """Return the union of all required controls across layers.

        Args:
            classifications: Previously computed layer classifications.

        Returns:
            De-duplicated list of all required safety controls.
        """
        controls: set[str] = set()
        for lc in classifications:
            controls.update(lc.safety_control.required_controls)
        return sorted(controls)
