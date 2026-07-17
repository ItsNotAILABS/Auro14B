"""Route tasks to potential-succotash engines/models the main Auro LLM can use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from auro_native_llm.succotash.registry import SuccotashRegistry, load_registry


@dataclass
class RouteResult:
    task: str
    model: Optional[Dict[str, Any]] = None
    engine: Optional[Dict[str, Any]] = None
    agent: Optional[Dict[str, Any]] = None
    protocol: Optional[Dict[str, str]] = None
    auro_lane: str = "Auro-2B"
    rationale: str = ""
    compute_plane: str = "MESIE"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.succotash.route.v1",
            "task": self.task,
            "model": self.model,
            "engine": self.engine,
            "agent": self.agent,
            "protocol": self.protocol,
            "auro_lane": self.auro_lane,
            "rationale": self.rationale,
            "compute_plane": self.compute_plane,
            "score": self.score,
        }


@dataclass
class EngineModelRouter:
    """Main LLM uses this to pick engines/models/agents from succotash."""

    registry: SuccotashRegistry
    prefer_native: bool = True

    @classmethod
    def load(cls, **kwargs: Any) -> "EngineModelRouter":
        return cls(registry=load_registry(**kwargs))

    def list_models(self) -> List[Dict[str, Any]]:
        return self.registry.models_for_llm()

    def list_engines(self) -> List[Dict[str, Any]]:
        return self.registry.engines_for_llm()

    def list_agents(self) -> List[Dict[str, Any]]:
        return self.registry.agents_for_llm()

    def route(self, task: str, *, intent: str = "") -> RouteResult:
        text = f"{task} {intent}".lower()
        result = RouteResult(task=task)

        # Agent keywords
        agent_map = {
            "research": "Researcher",
            "crawl": "Crawler",
            "scrape": "Scraper",
            "scout": "Scout",
            "digest": "Digest",
            "monitor": "Monitor",
            "analy": "Analyst",
            "sweep": "Sweep",
            "corpus": "Corpus",
            "memory": "Memoria",
            "sense": "Sensus",
        }
        for key, name in agent_map.items():
            if key in text:
                for a in self.registry.agents:
                    if a.name.lower() == name.lower():
                        result.agent = a.to_dict()
                        break
                break

        # Engine keywords
        engine_map = [
            (("summar", "classify", "offline", "solus", "qa"), "Solus"),
            (("memory", "palace", "remember"), "Memory Palace"),
            (("phish", "sentry", "threat", "inject"), "SentryAI"),
            (("graph", "entity", "cartograph"), "Cartographer"),
            (("crawl", "fetch", "spider"), "CrawlFetcher"),
            (("route", "dispatch", "orchestr"), "ModelRouter"),
            (("spectral", "mesie", "helix"), "MESIE SpectralGPT"),
            (("document", "absorb", "ingest"), "DocumentAbsorption"),
            (("heartbeat", "neuro", "phi"), "NeuroCore"),
            (("pattern", "primitive"), "PatternSynthesis"),
        ]
        for keys, name in engine_map:
            if any(k in text for k in keys):
                for e in self.registry.engines:
                    if e.name.lower() == name.lower():
                        result.engine = e.to_dict()
                        break
                break
        if result.engine is None and self.registry.engines:
            # default native MESIE
            for e in self.registry.engines:
                if "MESIE" in e.name:
                    result.engine = e.to_dict()
                    break

        # Model family routing (capability match)
        best = None
        best_score = -1.0
        for m in self.registry.model_families:
            blob = " ".join(
                [
                    m.family_name,
                    m.alpha_model,
                    m.primary_capability,
                    m.secondary_capabilities,
                    m.intelligence_class,
                    m.modality,
                ]
            ).lower()
            score = 0.0
            for tok in text.split():
                if len(tok) > 3 and tok in blob:
                    score += 1.0
            if "code" in text and "code" in blob:
                score += 2.0
            if "vision" in text or "image" in text:
                if "vision" in blob or "image" in blob:
                    score += 2.0
            if "rag" in text or "retriev" in text:
                if "rag" in blob or "retriev" in blob:
                    score += 2.0
            if "edge" in text and "edge" in blob:
                score += 1.5
            if "moe" in text and "moe" in blob:
                score += 1.5
            if score > best_score:
                best_score = score
                best = m
        if best and best_score > 0:
            result.model = best.to_dict()
            result.score = best_score
        elif self.prefer_native:
            result.model = {
                "family_id": "AURO-2B",
                "family_name": "Auro-2B",
                "alpha_model": "Auro-2B",
                "native": True,
                "compute_plane": "MESIE",
                "wire_protocol": "mesie.foundation.SpectralGPT",
            }
            result.score = 0.5

        # Protocol
        for p in self.registry.protocols:
            name = (p.get("protocol_name") or "").lower()
            fn = (p.get("primary_function") or "").lower()
            if any(t in name or t in fn for t in text.split() if len(t) > 4):
                result.protocol = p
                break

        # Auro lane scaling
        if any(k in text for k in ("orchestr", "plan", "multi-agent", "frontier")):
            result.auro_lane = "Auro-14B"
        elif any(k in text for k in ("reason", "general", "code")):
            result.auro_lane = "Auro-8B"
        elif any(k in text for k in ("special", "edit", "match")):
            result.auro_lane = "Auro-4B"
        else:
            result.auro_lane = "Auro-2B"

        parts = []
        if result.model:
            parts.append(f"model={result.model.get('family_name') or result.model.get('alpha_model')}")
        if result.engine:
            parts.append(f"engine={result.engine.get('name')}")
        if result.agent:
            parts.append(f"agent={result.agent.get('name')}")
        parts.append(f"lane={result.auro_lane}")
        result.rationale = "; ".join(parts)
        result.compute_plane = "MESIE" if self.prefer_native else "mixed"
        return result

    def catalogue_prompt(self, max_models: int = 12, max_engines: int = 10) -> str:
        """Text block the main LLM can absorb as system knowledge."""
        lines = [
            "SUCCOTASH ENGINE/MODEL CATALOGUE (FreddyCreates/potential-succotash)",
            "Native compute plane remains MESIE SpectralGPT; external families are routing targets.",
            "",
            "MODELS:",
        ]
        for m in self.list_models()[:max_models]:
            lines.append(
                f"  - {m.get('family_id')} {m.get('family_name')}: "
                f"{m.get('primary_capability')} [{m.get('routing_priority')}]"
            )
        lines.append("ENGINES:")
        for e in self.list_engines()[:max_engines]:
            lines.append(f"  - {e.get('engine_id')} {e.get('name')} ({e.get('kind')})")
        lines.append("AGENTS:")
        for a in self.list_agents()[:12]:
            lines.append(f"  - {a.get('agent_id')} {a.get('name')} ({a.get('role')})")
        return "\n".join(lines)


def route_task(task: str, **kwargs: Any) -> Dict[str, Any]:
    return EngineModelRouter.load().route(task, **kwargs).to_dict()
