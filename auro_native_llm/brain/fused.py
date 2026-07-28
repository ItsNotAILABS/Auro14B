"""A dependency-free fusion of HIM and MESIE's useful cognitive mechanisms.

This is an inference-time cognitive substrate, not a claim that biological
simulation changes the checkpoint's learned parameter count.  It provides
persistent state, bounded activation propagation, salience, working-memory
gating, arbitration and tamper-evident cycle receipts around the open model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True)
class BrainRegion:
    abbreviation: str
    system: str
    role: str


@dataclass(frozen=True)
class BrainCycle:
    cycle: int
    salience: float
    coherence: float
    anomaly: float
    dominant_system: str
    route: str
    working_memory: tuple[str, ...]
    receipt_hash: str


# Stable public topology extracted from MESIE's 44-region connectome.  The
# canonical representation intentionally has no numpy or mesie dependency.
_TOPOLOGY = (
 ("DLPFC_L","prefrontal","working_memory_executive_control"),("DLPFC_R","prefrontal","attention_task_switching"),
 ("vmPFC","prefrontal","value_assessment_reward_prediction"),("OFC","prefrontal","reward_evaluation_decision_making"),
 ("ACC","prefrontal","conflict_monitoring_error_detection"),("M1_L","motor","action_execution_output"),
 ("M1_R","motor","action_execution_output"),("SMA","motor","action_planning_sequencing"),("PMC","motor","movement_preparation"),
 ("S1_L","somatosensory","sensory_input_processing"),("S1_R","somatosensory","sensory_input_processing"),
 ("STG_L","temporal","language_comprehension_auditory"),("STG_R","temporal","prosody_processing"),
 ("ITG_L","temporal","object_recognition_visual_memory"),("WER","temporal","language_understanding_semantics"),
 ("BRO","temporal","language_production_syntax"),("PPC_L","parietal","spatial_reasoning_attention"),
 ("PPC_R","parietal","spatial_integration_navigation"),("AG","parietal","semantic_integration_abstraction"),
 ("SMG","parietal","phonological_processing"),("V1","occipital","visual_input_feature_extraction"),
 ("V2V3","occipital","visual_pattern_recognition"),("FFA","occipital","identity_recognition"),
 ("HPC_L","limbic","episodic_memory_encoding"),("HPC_R","limbic","memory_consolidation"),
 ("AMY_L","limbic","threat_detection"),("AMY_R","limbic","emotional_salience"),("INS_L","limbic","interoception"),
 ("PCC","limbic","self_referential_context"),("THL_L","subcortical","sensory_relay_gating"),
 ("THL_R","subcortical","sensory_relay_gating"),("CAU","subcortical","goal_directed_learning"),
 ("PUT","subcortical","reinforcement_learning"),("NAc","subcortical","reward_motivation"),
 ("GP","subcortical","action_selection_inhibition"),("HYP","subcortical","homeostasis_drive_regulation"),
 ("CBV","cerebellar","timing_coordination_prediction"),("CBH_L","cerebellar","prediction_error_correction"),
 ("CBH_R","cerebellar","cognitive_prediction_modeling"),("SC","brainstem","orienting_attention"),
 ("PON","brainstem","arousal_regulation"),("MED","brainstem","autonomic_regulation"),
 ("LC","brainstem","alertness_focus"),("VTA","brainstem","reward_signal"),
)

_KEYWORDS = {
 "prefrontal": ("plan","reason","decide","logic","compare"), "temporal": ("text","language","remember","meaning"),
 "limbic": ("risk","urgent","important","identity","memory"), "motor": ("run","build","write","execute","create"),
 "parietal": ("integrate","graph","spatial","connect"), "occipital": ("image","visual","see","browser"),
 "subcortical": ("reward","select","gate","goal"), "cerebellar": ("predict","verify","correct","timing"),
 "brainstem": ("monitor","alert","security","health"), "somatosensory": ("input","signal","data","observe"),
}


class HIMBrain:
    """Stateful cognitive controller shared by HIM inference and agents."""
    schema = "him.brain.v1"

    def __init__(self, state_path: str | Path | None = None, memory_limit: int = 8):
        self.regions = tuple(BrainRegion(*row) for row in _TOPOLOGY)
        self.state_path = Path(state_path) if state_path else None
        self.memory_limit = memory_limit
        self.cycle_number = 0
        self.activations = {r.abbreviation: 0.0 for r in self.regions}
        self.working_memory: list[str] = []
        self.previous_hash = "0" * 64
        if self.state_path and self.state_path.exists(): self._load()

    def cycle(self, observation: str, *, importance: float = 0.5, execute_requested: bool = False) -> BrainCycle:
        text = str(observation).strip(); lower = text.lower(); importance = _bound(importance)
        system_scores = {system: sum(1 for word in words if word in lower) for system, words in _KEYWORDS.items()}
        if not any(system_scores.values()): system_scores["prefrontal"] = 1
        peak = max(system_scores.values()) or 1
        for region in self.regions:
            stimulus = system_scores[region.system] / peak
            recurrent = self.activations[region.abbreviation] * .64
            self.activations[region.abbreviation] = _bound(recurrent + stimulus * .28 + importance * .08)
        values = list(self.activations.values()); mean = sum(values) / len(values)
        variance = sum((v-mean)**2 for v in values) / len(values)
        coherence = _bound(1.0 - math.sqrt(variance)); anomaly = _bound((1.0-coherence) * .65 + importance * .35)
        dominant = max(system_scores, key=system_scores.get)
        salience = _bound(.45 * importance + .35 * max(values) + .20 * anomaly)
        route = "execute" if execute_requested and salience >= .55 else ("deliberate" if salience >= .35 else "answer")
        if text and (salience >= .35 or importance >= .5):
            self.working_memory = (self.working_memory + [text[:500]])[-self.memory_limit:]
        self.cycle_number += 1
        payload = {"cycle":self.cycle_number,"salience":salience,"coherence":coherence,"anomaly":anomaly,
                   "dominant_system":dominant,"route":route,"working_memory":self.working_memory,"previous":self.previous_hash}
        receipt = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.previous_hash = receipt; result = BrainCycle(self.cycle_number,salience,coherence,anomaly,dominant,route,tuple(self.working_memory),receipt)
        self._save(); return result

    def snapshot(self) -> dict[str, Any]:
        systems: dict[str,float] = {}
        for region in self.regions: systems.setdefault(region.system,0.0); systems[region.system] += self.activations[region.abbreviation]
        counts = {s:sum(r.system==s for r in self.regions) for s in systems}
        return {"schema":self.schema,"cycle":self.cycle_number,"region_count":len(self.regions),"topology":[asdict(r) for r in self.regions],
                "systems":{s:round(v/counts[s],6) for s,v in systems.items()},"activations":dict(self.activations),
                "working_memory":list(self.working_memory),"receipt_head":self.previous_hash,"legacy_adapter":"optional"}

    state = snapshot  # stable alias used by the earlier BRAIN/MESIE surface

    def legacy_parity(self) -> dict[str, Any]:
        try:
            from mesie.connectome.brain_regions import get_default_regions
            legacy = [r.abbreviation for r in get_default_regions()]
            canonical = [r.abbreviation for r in self.regions]
            return {"available":True,"match":legacy==canonical,"legacy_count":len(legacy),"canonical_count":len(canonical)}
        except (ImportError, ModuleNotFoundError):
            return {"available":False,"match":None,"canonical_count":len(self.regions),"reason":"MESIE not installed; canonical brain is independent"}

    def _save(self):
        if not self.state_path: return
        self.state_path.parent.mkdir(parents=True,exist_ok=True)
        body={"schema":self.schema,"cycle":self.cycle_number,"activations":self.activations,"working_memory":self.working_memory,
              "previous_hash":self.previous_hash,"saved_at":time.time()}
        temp=self.state_path.with_suffix(self.state_path.suffix+".tmp"); temp.write_text(json.dumps(body,sort_keys=True),encoding="utf-8"); temp.replace(self.state_path)

    def _load(self):
        body=json.loads(self.state_path.read_text(encoding="utf-8")); self.cycle_number=int(body.get("cycle",0))
        for key,value in body.get("activations",{}).items():
            if key in self.activations: self.activations[key]=_bound(value)
        self.working_memory=[str(x)[:500] for x in body.get("working_memory",[])][-self.memory_limit:]
        self.previous_hash=str(body.get("previous_hash",self.previous_hash))


def _bound(value: Any) -> float:
    try: return max(0.0,min(1.0,float(value)))
    except (TypeError,ValueError): return 0.0
