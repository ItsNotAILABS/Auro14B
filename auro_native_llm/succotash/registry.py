"""Load potential-succotash registers: models, engines, agents, protocols, extensions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from auro_native_llm.succotash.paths import ensure_succotash, succotash_root


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        # utf-8-sig for BOM
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows: List[Dict[str, str]] = []
        for row in reader:
            clean = {str(k).strip(): (v or "").strip() for k, v in row.items() if k}
            if any(clean.values()):
                rows.append(clean)
        return rows


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


@dataclass
class ModelFamily:
    family_id: str
    family_name: str
    alpha_model: str = ""
    parent_org: str = ""
    intelligence_class: str = ""
    primary_capability: str = ""
    secondary_capabilities: str = ""
    parameter_class: str = ""
    context_window: str = ""
    modality: str = ""
    ring_affinity: str = ""
    organism_placement: str = ""
    routing_priority: str = ""
    wire_protocol: str = ""
    engine_status: str = "active"
    raw: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_name": self.family_name,
            "alpha_model": self.alpha_model,
            "parent_org": self.parent_org,
            "intelligence_class": self.intelligence_class,
            "primary_capability": self.primary_capability,
            "secondary_capabilities": self.secondary_capabilities,
            "parameter_class": self.parameter_class,
            "context_window": self.context_window,
            "modality": self.modality,
            "ring_affinity": self.ring_affinity,
            "organism_placement": self.organism_placement,
            "routing_priority": self.routing_priority,
            "wire_protocol": self.wire_protocol,
            "engine_status": self.engine_status,
        }


@dataclass
class EngineSpec:
    engine_id: str
    name: str
    kind: str  # compute | routing | memory | agent | protocol | solus | sdk
    path: str = ""
    description: str = ""
    wire_protocol: str = ""
    status: str = "active"
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "description": self.description,
            "wire_protocol": self.wire_protocol,
            "status": self.status,
        }


@dataclass
class AgentSpec:
    agent_id: str
    name: str
    path: str = ""
    role: str = ""
    description: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "path": self.path,
            "role": self.role,
            "description": self.description,
        }


@dataclass
class SuccotashRegistry:
    """Unified engines + models catalogue from potential-succotash."""

    root: str = ""
    source_url: str = "https://github.com/FreddyCreates/potential-succotash"
    model_families: List[ModelFamily] = field(default_factory=list)
    multimodal_families: List[Dict[str, str]] = field(default_factory=list)
    protocols: List[Dict[str, str]] = field(default_factory=list)
    extensions: List[Dict[str, str]] = field(default_factory=list)
    architectural_laws: List[Dict[str, str]] = field(default_factory=list)
    frontier_models: List[Dict[str, str]] = field(default_factory=list)
    marketplace: List[Dict[str, str]] = field(default_factory=list)
    phantom_blockchain: List[Dict[str, str]] = field(default_factory=list)
    sdk_manifest: Dict[str, Any] = field(default_factory=dict)
    engines: List[EngineSpec] = field(default_factory=list)
    agents: List[AgentSpec] = field(default_factory=list)
    sdks: List[str] = field(default_factory=list)
    training_areas: List[str] = field(default_factory=list)

    def models_for_llm(self) -> List[Dict[str, Any]]:
        """Models the main LLM can list / route to."""
        out = [m.to_dict() for m in self.model_families if m.engine_status.lower() != "retired"]
        # Auro family always present as native MESIE cores
        for mid, tier in (
            ("Auro-2B", "edge"),
            ("Auro-4B", "specialist"),
            ("Auro-8B", "general"),
            ("Auro-14B", "orchestrator"),
            ("Auro-100B", "frontier"),
        ):
            out.append(
                {
                    "family_id": f"AURO-{mid.split('-')[1]}",
                    "family_name": mid,
                    "alpha_model": mid,
                    "parent_org": "Auro/MESIE",
                    "intelligence_class": "MESIE SpectralGPT MoE",
                    "primary_capability": "native sovereign text + spectral",
                    "parameter_class": f"target {mid}",
                    "modality": "Text + Spectral",
                    "ring_affinity": "Sovereign Ring",
                    "organism_placement": f"Auro mind / {tier}",
                    "routing_priority": "P0 — native",
                    "wire_protocol": "mesie.foundation.SpectralGPT",
                    "engine_status": "active",
                    "native": True,
                    "compute_plane": "MESIE",
                }
            )
        return out

    def engines_for_llm(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.engines]

    def agents_for_llm(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.agents]

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source_url,
            "root": self.root,
            "model_families": len(self.model_families),
            "multimodal_families": len(self.multimodal_families),
            "protocols": len(self.protocols),
            "extensions": len(self.extensions),
            "architectural_laws": len(self.architectural_laws),
            "frontier_models": len(self.frontier_models),
            "marketplace": len(self.marketplace),
            "engines": len(self.engines),
            "agents": len(self.agents),
            "sdks": len(self.sdks),
            "training_areas": self.training_areas,
            "sdk_modules": sum(
                len(s.get("modules") or [])
                for s in (self.sdk_manifest.get("sdks") or [])
                if isinstance(s, dict)
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.succotash.registry.v1",
            "summary": self.summary(),
            "models": self.models_for_llm(),
            "engines": self.engines_for_llm(),
            "agents": self.agents_for_llm(),
            "protocols": self.protocols[:50],
            "extensions": self.extensions[:50],
            "multimodal_families": self.multimodal_families,
            "sdks": self.sdks,
            "training_areas": self.training_areas,
        }

    def training_lexicon(self, max_items: int = 400) -> List[str]:
        """Words / capability phrases for tokenizer + meaning training."""
        words: List[str] = []
        seen: set[str] = set()

        def add(s: str) -> None:
            s = (s or "").strip()
            if not s or s in seen:
                return
            seen.add(s)
            words.append(s)

        # Core organism vocabulary first (always present for tokenizer/meaning)
        for w in (
            "MESIE", "SpectralGPT", "Solus", "NeuroCore", "Memory Palace",
            "phi", "golden ratio", "sovereign", "organism", "receipt",
            "Auro-2B", "Auro-4B", "Auro-8B", "Auro-14B", "Auro-100B",
            "Sonic Ninja", "Vigil", "SentryAI", "Cartographer", "CrawlFetcher",
            "PatternSynthesis", "intelligence-wire", "ring affinity",
            "potential-succotash", "FreddyCreates", "Medina",
        ):
            add(w)

        for m in self.model_families:
            add(m.family_name)
            add(m.alpha_model)
            add(m.intelligence_class)
            add(m.primary_capability)
            for part in (m.secondary_capabilities or "").replace("/", ",").split(","):
                add(part.strip())
            add(m.ring_affinity)
            add(m.organism_placement)
        for p in self.protocols:
            add(p.get("protocol_name", ""))
            add(p.get("primary_function", ""))
            add(p.get("intelligence_class", ""))
        for e in self.extensions:
            add(e.get("extension_name", ""))
            add(e.get("primary_capability", ""))
        for a in self.agents:
            add(a.name)
            add(a.role)
        for eng in self.engines:
            add(eng.name)
            add(eng.kind)
        return words[:max_items]


# Builtin engine inventory from succotash README + sdk layout
_BUILTIN_ENGINES = [
    ("ENG-SOLUS", "Solus", "solus", "sdk/", "Offline Transformers.js summarize/classify/QA"),
    ("ENG-NEURO", "NeuroCore", "compute", "sdk/", "Phi-encoded 873ms heartbeat oscillator"),
    ("ENG-PATTERN", "PatternSynthesis", "compute", "sdk/", "40 cognitive primitives / 8 domains"),
    ("ENG-MEMORY", "Memory Palace", "memory", "memory_temple/", "5D phi spatial memory"),
    ("ENG-SENTRY", "SentryAI", "security", "defense-organism/", "Phishing/PII/injection defense"),
    ("ENG-CARTO", "Cartographer", "knowledge", "sdk/", "Knowledge graph + entity extraction"),
    ("ENG-CRAWL", "CrawlFetcher", "agent", "sdk/agents/", "Parallel agent fetch/scrape"),
    ("ENG-MESIE", "MESIE SpectralGPT", "compute", "mesie", "Auro native transformer MoE"),
    ("ENG-ROUTER", "ModelRouter", "routing", "sdk/intelligence-routing-sdk/", "Task→model routing"),
    ("ENG-ALPHA", "AlphaResolver", "routing", "sdk/ai-model-engines/", "Alpha model resolution"),
    ("ENG-CORE", "EngineCore", "compute", "sdk/ai-model-engines/", "AI model pipeline core"),
    ("ENG-WIRE", "ModelWire", "routing", "sdk/ai-model-engines/", "Model wire protocol"),
    ("ENG-DOC", "DocumentAbsorption", "knowledge", "sdk/document-absorption-engine/", "Doc parse + ingest"),
    ("ENG-CHRONO", "ChronoEngine", "compute", "sdk/engines/", "Temporal engine"),
    ("ENG-NEXORIS", "NexorisEngine", "compute", "sdk/engines/", "Nexus compute engine"),
    ("ENG-QUANTUM", "QuantumFluxEngine", "compute", "sdk/engines/", "Quantum-flux engine"),
    ("ENG-COREO", "CoreographEngine", "compute", "sdk/engines/", "Coreography engine"),
]

_BUILTIN_AGENTS = [
    ("AGT-RESEARCHER", "Researcher", "research", "Wikipedia + domain parallel synthesis"),
    ("AGT-CRAWLER", "Crawler", "crawl", "Spider URL, follow links, extract"),
    ("AGT-SCRAPER", "Scraper", "extract", "Tables/lists/prices → structured"),
    ("AGT-SCOUT", "Scout", "scan", "Deep scan + full link map"),
    ("AGT-DIGEST", "Digest", "report", "Multi-topic synthesis reports"),
    ("AGT-MONITOR", "Monitor", "watch", "Watch sites for changes"),
    ("AGT-ANALYST", "Analyst", "compare", "Parallel multi-URL comparison"),
    ("AGT-SWEEP", "Sweep", "batch", "Batch extraction across sites"),
    ("AGT-ANIMUS", "Animus", "agent", "sdk/agents/animus-agent.js"),
    ("AGT-CORPUS", "Corpus", "agent", "sdk/agents/corpus-agent.js"),
    ("AGT-MEMORIA", "Memoria", "agent", "sdk/agents/memoria-agent.js"),
    ("AGT-SENSUS", "Sensus", "agent", "sdk/agents/sensus-agent.js"),
]


def load_registry(*, clone: bool = True) -> SuccotashRegistry:
    """Load full potential-succotash registry for LLM engines/models."""
    root = ensure_succotash(clone=clone) if clone else succotash_root()
    if root is None:
        raise FileNotFoundError("potential-succotash not available")

    reg = SuccotashRegistry(root=str(root))

    # Model families
    for row in _read_csv(root / "AI_Model_Families_Register.csv"):
        reg.model_families.append(
            ModelFamily(
                family_id=row.get("family_id", ""),
                family_name=row.get("family_name", ""),
                alpha_model=row.get("alpha_model", ""),
                parent_org=row.get("parent_org", ""),
                intelligence_class=row.get("intelligence_class", ""),
                primary_capability=row.get("primary_capability", ""),
                secondary_capabilities=row.get("secondary_capabilities", ""),
                parameter_class=row.get("parameter_class", ""),
                context_window=row.get("context_window", ""),
                modality=row.get("modality", ""),
                ring_affinity=row.get("ring_affinity", ""),
                organism_placement=row.get("organism_placement", ""),
                routing_priority=row.get("routing_priority", ""),
                wire_protocol=row.get("wire_protocol", ""),
                engine_status=row.get("engine_status", "active") or "active",
                raw=row,
            )
        )

    reg.multimodal_families = _read_csv(root / "Multimodal_Families_Register.csv")
    reg.protocols = _read_csv(root / "AI_Protocols_Register.csv")
    reg.extensions = _read_csv(root / "AI_Extensions_Register.csv")
    reg.architectural_laws = _read_csv(root / "Architectural_Laws_Register.csv")
    reg.frontier_models = _read_csv(root / "Frontend_Frontier_100_Register.csv")
    reg.marketplace = _read_csv(root / "Organism_Marketplace_Register.csv")
    reg.phantom_blockchain = _read_csv(root / "Phantom_Blockchain_Model_Register.csv")

    manifest = _read_json(root / "SDK_Model_Manifest.json")
    if isinstance(manifest, dict):
        reg.sdk_manifest = manifest

    # SDKs on disk
    sdk_dir = root / "sdk"
    if sdk_dir.exists():
        reg.sdks = sorted(
            p.name for p in sdk_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    # Engines
    for eid, name, kind, path, desc in _BUILTIN_ENGINES:
        reg.engines.append(
            EngineSpec(
                engine_id=eid,
                name=name,
                kind=kind,
                path=path,
                description=desc,
                wire_protocol=f"succotash/{kind}",
            )
        )
    # Protocol engines_wired fields
    for p in reg.protocols:
        wired = p.get("engines_wired", "")
        for piece in wired.replace("+", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            if any(e.name.lower() == piece.lower() for e in reg.engines):
                continue
            reg.engines.append(
                EngineSpec(
                    engine_id=f"ENG-{piece.upper().replace(' ', '-')[:24]}",
                    name=piece,
                    kind="protocol_engine",
                    description=f"Wired by {p.get('protocol_name', 'protocol')}",
                    wire_protocol=p.get("wire_protocol", ""),
                )
            )

    # Agents
    for aid, name, role, path in _BUILTIN_AGENTS:
        reg.agents.append(
            AgentSpec(agent_id=aid, name=name, role=role, path=path, description=path)
        )
    agents_dir = root / "sdk" / "agents"
    if agents_dir.exists():
        for p in agents_dir.glob("*-agent.js"):
            name = p.stem.replace("-agent", "").replace("_", " ").title()
            if any(a.name.lower() == name.lower() for a in reg.agents):
                continue
            reg.agents.append(
                AgentSpec(
                    agent_id=f"AGT-{p.stem.upper()}",
                    name=name,
                    path=str(p.relative_to(root)).replace("\\", "/"),
                    role="sdk_agent",
                )
            )

    # Training area names (directories that feed corpus)
    for area in (
        "docs", "research", "sdk", "protocols", "extensions", "organism",
        "governance", "architecture", "memory_temple", "defense-organism",
        "workers", "workflows", "production-grade-builder", "phantom_native",
        "examples", "atlas", "scripts",
    ):
        if (root / area).exists():
            reg.training_areas.append(area)
    # Root registers as synthetic "registers" area
    reg.training_areas.insert(0, "registers")

    return reg
