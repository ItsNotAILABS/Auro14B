"""Brain region definitions with anatomical 3D coordinates.

Defines real brain regions organized by functional systems, each with
approximate MNI-space 3D coordinates for spatial placement in the
simulated connectome environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class BrainSystem(Enum):
    """Major functional brain systems."""

    PREFRONTAL = "prefrontal"
    MOTOR = "motor"
    SOMATOSENSORY = "somatosensory"
    TEMPORAL = "temporal"
    PARIETAL = "parietal"
    OCCIPITAL = "occipital"
    LIMBIC = "limbic"
    SUBCORTICAL = "subcortical"
    CEREBELLAR = "cerebellar"
    BRAINSTEM = "brainstem"


@dataclass
class BrainRegion:
    """A single brain region with spatial and functional metadata.

    Attributes:
        name: Anatomical name of the region.
        abbreviation: Short code (e.g., 'DLPFC').
        system: Functional system this region belongs to.
        position_3d: (x, y, z) MNI-space centroid in mm.
        volume_mm3: Approximate volume in cubic mm.
        role: Functional role description for the AI backend.
        activation_level: Current simulated activation [0, 1].
        spectral_signature: Optional resonance fingerprint vector.
    """

    name: str
    abbreviation: str
    system: BrainSystem
    position_3d: Tuple[float, float, float]
    volume_mm3: float = 1000.0
    role: str = ""
    activation_level: float = 0.0
    spectral_signature: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def position_array(self) -> np.ndarray:
        """Return position as numpy array."""
        return np.array(self.position_3d, dtype=np.float64)


def get_default_regions() -> List[BrainRegion]:
    """Return a comprehensive set of real brain regions with 3D positions.

    Positions are approximate MNI coordinates (x, y, z) in mm.
    Regions are grouped into functional systems that map to AI backend roles.

    Returns:
        List of BrainRegion instances covering major cortical and
        subcortical structures.
    """
    regions = [
        # === PREFRONTAL CORTEX — Executive Control / Decision Making ===
        BrainRegion(
            name="Dorsolateral Prefrontal Cortex (L)",
            abbreviation="DLPFC_L",
            system=BrainSystem.PREFRONTAL,
            position_3d=(-40.0, 32.0, 30.0),
            volume_mm3=12500.0,
            role="working_memory_executive_control",
        ),
        BrainRegion(
            name="Dorsolateral Prefrontal Cortex (R)",
            abbreviation="DLPFC_R",
            system=BrainSystem.PREFRONTAL,
            position_3d=(40.0, 32.0, 30.0),
            volume_mm3=12500.0,
            role="attention_task_switching",
        ),
        BrainRegion(
            name="Ventromedial Prefrontal Cortex",
            abbreviation="vmPFC",
            system=BrainSystem.PREFRONTAL,
            position_3d=(0.0, 48.0, -12.0),
            volume_mm3=9800.0,
            role="value_assessment_reward_prediction",
        ),
        BrainRegion(
            name="Orbitofrontal Cortex",
            abbreviation="OFC",
            system=BrainSystem.PREFRONTAL,
            position_3d=(0.0, 38.0, -18.0),
            volume_mm3=11000.0,
            role="reward_evaluation_decision_making",
        ),
        BrainRegion(
            name="Anterior Cingulate Cortex",
            abbreviation="ACC",
            system=BrainSystem.PREFRONTAL,
            position_3d=(0.0, 30.0, 24.0),
            volume_mm3=8500.0,
            role="conflict_monitoring_error_detection",
        ),
        # === MOTOR CORTEX — Action Generation ===
        BrainRegion(
            name="Primary Motor Cortex (L)",
            abbreviation="M1_L",
            system=BrainSystem.MOTOR,
            position_3d=(-36.0, -20.0, 58.0),
            volume_mm3=10200.0,
            role="action_execution_output",
        ),
        BrainRegion(
            name="Primary Motor Cortex (R)",
            abbreviation="M1_R",
            system=BrainSystem.MOTOR,
            position_3d=(36.0, -20.0, 58.0),
            volume_mm3=10200.0,
            role="action_execution_output",
        ),
        BrainRegion(
            name="Supplementary Motor Area",
            abbreviation="SMA",
            system=BrainSystem.MOTOR,
            position_3d=(0.0, -8.0, 62.0),
            volume_mm3=7800.0,
            role="action_planning_sequencing",
        ),
        BrainRegion(
            name="Premotor Cortex",
            abbreviation="PMC",
            system=BrainSystem.MOTOR,
            position_3d=(-30.0, -4.0, 54.0),
            volume_mm3=9000.0,
            role="movement_preparation",
        ),
        # === SOMATOSENSORY — Input Processing ===
        BrainRegion(
            name="Primary Somatosensory Cortex (L)",
            abbreviation="S1_L",
            system=BrainSystem.SOMATOSENSORY,
            position_3d=(-42.0, -28.0, 54.0),
            volume_mm3=9500.0,
            role="sensory_input_processing",
        ),
        BrainRegion(
            name="Primary Somatosensory Cortex (R)",
            abbreviation="S1_R",
            system=BrainSystem.SOMATOSENSORY,
            position_3d=(42.0, -28.0, 54.0),
            volume_mm3=9500.0,
            role="sensory_input_processing",
        ),
        # === TEMPORAL — Language / Memory / Audio ===
        BrainRegion(
            name="Superior Temporal Gyrus (L)",
            abbreviation="STG_L",
            system=BrainSystem.TEMPORAL,
            position_3d=(-56.0, -22.0, 6.0),
            volume_mm3=14000.0,
            role="language_comprehension_auditory",
        ),
        BrainRegion(
            name="Superior Temporal Gyrus (R)",
            abbreviation="STG_R",
            system=BrainSystem.TEMPORAL,
            position_3d=(56.0, -22.0, 6.0),
            volume_mm3=14000.0,
            role="prosody_music_processing",
        ),
        BrainRegion(
            name="Inferior Temporal Gyrus (L)",
            abbreviation="ITG_L",
            system=BrainSystem.TEMPORAL,
            position_3d=(-50.0, -40.0, -20.0),
            volume_mm3=11500.0,
            role="object_recognition_visual_memory",
        ),
        BrainRegion(
            name="Wernicke's Area",
            abbreviation="WER",
            system=BrainSystem.TEMPORAL,
            position_3d=(-58.0, -40.0, 12.0),
            volume_mm3=6200.0,
            role="language_understanding_semantics",
        ),
        BrainRegion(
            name="Broca's Area",
            abbreviation="BRO",
            system=BrainSystem.TEMPORAL,
            position_3d=(-48.0, 18.0, 12.0),
            volume_mm3=5800.0,
            role="language_production_syntax",
        ),
        # === PARIETAL — Integration / Spatial ===
        BrainRegion(
            name="Posterior Parietal Cortex (L)",
            abbreviation="PPC_L",
            system=BrainSystem.PARIETAL,
            position_3d=(-36.0, -58.0, 48.0),
            volume_mm3=13000.0,
            role="spatial_reasoning_attention",
        ),
        BrainRegion(
            name="Posterior Parietal Cortex (R)",
            abbreviation="PPC_R",
            system=BrainSystem.PARIETAL,
            position_3d=(36.0, -58.0, 48.0),
            volume_mm3=13000.0,
            role="spatial_integration_navigation",
        ),
        BrainRegion(
            name="Angular Gyrus",
            abbreviation="AG",
            system=BrainSystem.PARIETAL,
            position_3d=(-44.0, -64.0, 30.0),
            volume_mm3=7200.0,
            role="semantic_integration_abstraction",
        ),
        BrainRegion(
            name="Supramarginal Gyrus",
            abbreviation="SMG",
            system=BrainSystem.PARIETAL,
            position_3d=(-52.0, -40.0, 34.0),
            volume_mm3=6800.0,
            role="phonological_processing",
        ),
        # === OCCIPITAL — Visual Processing ===
        BrainRegion(
            name="Primary Visual Cortex (V1)",
            abbreviation="V1",
            system=BrainSystem.OCCIPITAL,
            position_3d=(0.0, -84.0, 4.0),
            volume_mm3=15000.0,
            role="visual_input_feature_extraction",
        ),
        BrainRegion(
            name="Visual Association Cortex (V2/V3)",
            abbreviation="V2V3",
            system=BrainSystem.OCCIPITAL,
            position_3d=(0.0, -76.0, 12.0),
            volume_mm3=12000.0,
            role="visual_pattern_recognition",
        ),
        BrainRegion(
            name="Fusiform Face Area",
            abbreviation="FFA",
            system=BrainSystem.OCCIPITAL,
            position_3d=(-40.0, -54.0, -18.0),
            volume_mm3=4500.0,
            role="face_recognition_identity",
        ),
        # === LIMBIC — Emotion / Memory Formation ===
        BrainRegion(
            name="Hippocampus (L)",
            abbreviation="HPC_L",
            system=BrainSystem.LIMBIC,
            position_3d=(-28.0, -20.0, -14.0),
            volume_mm3=3500.0,
            role="episodic_memory_encoding",
        ),
        BrainRegion(
            name="Hippocampus (R)",
            abbreviation="HPC_R",
            system=BrainSystem.LIMBIC,
            position_3d=(28.0, -20.0, -14.0),
            volume_mm3=3500.0,
            role="spatial_memory_consolidation",
        ),
        BrainRegion(
            name="Amygdala (L)",
            abbreviation="AMY_L",
            system=BrainSystem.LIMBIC,
            position_3d=(-22.0, -4.0, -18.0),
            volume_mm3=1800.0,
            role="emotional_valence_threat_detection",
        ),
        BrainRegion(
            name="Amygdala (R)",
            abbreviation="AMY_R",
            system=BrainSystem.LIMBIC,
            position_3d=(22.0, -4.0, -18.0),
            volume_mm3=1800.0,
            role="emotional_salience_fear_conditioning",
        ),
        BrainRegion(
            name="Insula (L)",
            abbreviation="INS_L",
            system=BrainSystem.LIMBIC,
            position_3d=(-38.0, 4.0, 2.0),
            volume_mm3=8500.0,
            role="interoception_embodied_awareness",
        ),
        BrainRegion(
            name="Cingulate Gyrus (Posterior)",
            abbreviation="PCC",
            system=BrainSystem.LIMBIC,
            position_3d=(0.0, -44.0, 30.0),
            volume_mm3=6000.0,
            role="self_referential_default_mode",
        ),
        # === SUBCORTICAL — Core Relay / Reward ===
        BrainRegion(
            name="Thalamus (L)",
            abbreviation="THL_L",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(-10.0, -18.0, 6.0),
            volume_mm3=6000.0,
            role="sensory_relay_gating",
        ),
        BrainRegion(
            name="Thalamus (R)",
            abbreviation="THL_R",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(10.0, -18.0, 6.0),
            volume_mm3=6000.0,
            role="sensory_relay_gating",
        ),
        BrainRegion(
            name="Caudate Nucleus",
            abbreviation="CAU",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(-12.0, 10.0, 10.0),
            volume_mm3=4200.0,
            role="habit_learning_goal_directed",
        ),
        BrainRegion(
            name="Putamen",
            abbreviation="PUT",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(-24.0, 4.0, 2.0),
            volume_mm3=5500.0,
            role="motor_learning_reinforcement",
        ),
        BrainRegion(
            name="Nucleus Accumbens",
            abbreviation="NAc",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(0.0, 10.0, -8.0),
            volume_mm3=1200.0,
            role="reward_motivation_dopamine",
        ),
        BrainRegion(
            name="Globus Pallidus",
            abbreviation="GP",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(-18.0, -2.0, 0.0),
            volume_mm3=2800.0,
            role="action_selection_inhibition",
        ),
        BrainRegion(
            name="Hypothalamus",
            abbreviation="HYP",
            system=BrainSystem.SUBCORTICAL,
            position_3d=(0.0, -4.0, -10.0),
            volume_mm3=1600.0,
            role="homeostasis_drive_regulation",
        ),
        # === CEREBELLAR — Timing / Coordination / Prediction ===
        BrainRegion(
            name="Cerebellar Vermis",
            abbreviation="CBV",
            system=BrainSystem.CEREBELLAR,
            position_3d=(0.0, -66.0, -30.0),
            volume_mm3=18000.0,
            role="timing_coordination_prediction",
        ),
        BrainRegion(
            name="Cerebellar Hemisphere (L)",
            abbreviation="CBH_L",
            system=BrainSystem.CEREBELLAR,
            position_3d=(-30.0, -66.0, -34.0),
            volume_mm3=22000.0,
            role="motor_prediction_error_correction",
        ),
        BrainRegion(
            name="Cerebellar Hemisphere (R)",
            abbreviation="CBH_R",
            system=BrainSystem.CEREBELLAR,
            position_3d=(30.0, -66.0, -34.0),
            volume_mm3=22000.0,
            role="cognitive_prediction_modeling",
        ),
        # === BRAINSTEM — Arousal / Core Regulation ===
        BrainRegion(
            name="Midbrain (Superior Colliculus)",
            abbreviation="SC",
            system=BrainSystem.BRAINSTEM,
            position_3d=(0.0, -30.0, -8.0),
            volume_mm3=2000.0,
            role="orienting_attention_saccades",
        ),
        BrainRegion(
            name="Pons",
            abbreviation="PON",
            system=BrainSystem.BRAINSTEM,
            position_3d=(0.0, -32.0, -28.0),
            volume_mm3=4500.0,
            role="arousal_sleep_wake_regulation",
        ),
        BrainRegion(
            name="Medulla Oblongata",
            abbreviation="MED",
            system=BrainSystem.BRAINSTEM,
            position_3d=(0.0, -36.0, -42.0),
            volume_mm3=3800.0,
            role="autonomic_vital_functions",
        ),
        BrainRegion(
            name="Locus Coeruleus",
            abbreviation="LC",
            system=BrainSystem.BRAINSTEM,
            position_3d=(0.0, -34.0, -24.0),
            volume_mm3=300.0,
            role="norepinephrine_alertness_focus",
        ),
        BrainRegion(
            name="Ventral Tegmental Area",
            abbreviation="VTA",
            system=BrainSystem.BRAINSTEM,
            position_3d=(0.0, -20.0, -14.0),
            volume_mm3=500.0,
            role="dopamine_reward_signal",
        ),
    ]
    return regions


def get_regions_by_system(
    system: BrainSystem,
    regions: Optional[List[BrainRegion]] = None,
) -> List[BrainRegion]:
    """Filter regions by brain system.

    Args:
        system: The BrainSystem to filter by.
        regions: Optional list; defaults to all default regions.

    Returns:
        List of BrainRegion belonging to the specified system.
    """
    if regions is None:
        regions = get_default_regions()
    return [r for r in regions if r.system == system]


def get_region_positions(
    regions: Optional[List[BrainRegion]] = None,
) -> np.ndarray:
    """Extract 3D position matrix from regions.

    Args:
        regions: Optional list; defaults to all default regions.

    Returns:
        Array of shape (N, 3) with region centroid positions.
    """
    if regions is None:
        regions = get_default_regions()
    return np.array([r.position_3d for r in regions], dtype=np.float64)
