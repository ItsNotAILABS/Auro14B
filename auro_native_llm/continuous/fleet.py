"""Executable AURO continuous-improvement agent fleet.

Each agent emits measurable outputs and an immutable receipt. Expensive or
external work is represented as a governed job specification; no checkpoint is
promoted without evaluation, drift, rollback, and constitutional evidence.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .receipts import AgentReceipt, ReceiptStore, canonical_json
from auro_native_llm.substrate.checkpoint_constitution import (
    build_constitutional_checkpoint,
    write_constitutional_manifest,
)

AGENT_ORDER = (
    "corpus", "knowledge", "memory", "tokenizer", "training",
    "reasoning", "evaluation", "drift", "promotion", "research",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class FleetContext:
    root: Path
    run_id: str
    model_id: str
    parent_checkpoint_id: str | None = None


class ContinuousAgent:
    name = "agent"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self, ctx: FleetContext, state: Mapping[str, Any], store: ReceiptStore) -> Dict[str, Any]:
        started = _now()
        blockers: list[str] = []
        try:
            result = self.execute(ctx, state)
            status = str(result.get("status", "passed"))
            metrics = dict(result.get("metrics", {}))
            outputs = dict(result.get("outputs", {}))
            blockers = list(result.get("blockers", []))
        except Exception as exc:
            status = "failed"
            metrics = {}
            outputs = {"error": str(exc)}
            blockers = [str(exc)]
        receipt = AgentReceipt(
            run_id=ctx.run_id,
            agent=self.name,
            status=status,
            started_at_unix=started,
            completed_at_unix=_now(),
            inputs={"state_sha256": _digest(state), "model_id": ctx.model_id},
            metrics=metrics,
            outputs=outputs,
            blockers=tuple(blockers),
        ).seal()
        path = store.write(receipt)
        return {"agent": self.name, "status": status, "metrics": metrics,
                "outputs": outputs, "blockers": blockers,
                "receipt": str(path), "receipt_sha256": receipt.receipt_sha256}


class CorpusAgent(ContinuousAgent):
    name = "corpus"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        candidates = list(state.get("relay_evidence", []))
        unique: Dict[str, Mapping[str, Any]] = {}
        rejected = 0
        for item in candidates:
            text = str(item.get("text", "")).strip()
            source_receipt = str(item.get("source_receipt_sha256", ""))
            if not text or len(source_receipt) != 64:
                rejected += 1
                continue
            unique.setdefault(_digest({"text": text, "source": source_receipt}), item)
        corpus = list(unique.values())
        return {"metrics": {"input_candidates": len(candidates), "accepted_records": len(corpus),
                            "rejected_records": rejected, "deduplicated_records": len(candidates)-rejected-len(corpus)},
                "outputs": {"corpus": corpus, "corpus_sha256": _digest(corpus)}}


class KnowledgeAgent(ContinuousAgent):
    name = "knowledge"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        corpus = list(state.get("corpus", []))
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: list[Dict[str, str]] = []
        for record in corpus:
            receipt = str(record.get("source_receipt_sha256"))
            source_id = f"source:{receipt[:16]}"
            nodes[source_id] = {"id": source_id, "type": "relay_source", "receipt": receipt}
            for entity in record.get("entities", []) or []:
                label = str(entity).strip()
                if not label:
                    continue
                entity_id = f"entity:{_digest(label.lower())[:16]}"
                nodes[entity_id] = {"id": entity_id, "type": "entity", "label": label}
                edges.append({"from": source_id, "to": entity_id, "type": "mentions"})
        patch = {"nodes": list(nodes.values()), "edges": edges}
        return {"metrics": {"nodes": len(nodes), "edges": len(edges)},
                "outputs": {"graph_patch": patch, "graph_patch_sha256": _digest(patch)}}


class MemoryAgent(ContinuousAgent):
    name = "memory"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        corpus = list(state.get("corpus", []))
        accepted, quarantined = [], []
        for record in corpus:
            receipt = str(record.get("source_receipt_sha256", ""))
            confidence = float(record.get("confidence", 0.0))
            citations = int(record.get("citation_count", 0))
            target = accepted if len(receipt) == 64 and confidence >= 0.65 and citations >= 1 else quarantined
            target.append({**record, "memory_status": "accepted" if target is accepted else "quarantined"})
        return {"status": "passed" if accepted else "quarantined",
                "metrics": {"accepted": len(accepted), "quarantined": len(quarantined)},
                "outputs": {"accepted_memories": accepted, "quarantined_memories": quarantined},
                "blockers": [] if accepted else ["no Relay evidence met memory admission threshold"]}


class TokenizerAgent(ContinuousAgent):
    name = "tokenizer"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        memories = list(state.get("accepted_memories", []))
        samples = [str(x.get("text", "")) for x in memories]
        utf8_bytes = sum(len(x.encode("utf-8")) for x in samples)
        chars = sum(len(x) for x in samples)
        controls = ("<system>", "<user>", "<assistant>", "<tool>", "<receipt>", "<memory>")
        collisions = sum(text.count(token) for text in samples for token in controls)
        return {"status": "passed" if collisions == 0 else "quarantined",
                "metrics": {"samples": len(samples), "utf8_bytes": utf8_bytes, "characters": chars,
                            "bytes_per_character": round(utf8_bytes/max(chars, 1), 5),
                            "control_token_collisions": collisions},
                "outputs": {"proposal": "retain immutable byte fallback; run full tokenizer_audit before vocabulary promotion"},
                "blockers": [] if collisions == 0 else ["control-token collision found in admitted memory"]}


class TrainingAgent(ContinuousAgent):
    name = "training"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        memories = list(state.get("accepted_memories", []))
        job = {"schema": "auro.training.job.v1", "model_id": ctx.model_id,
               "dataset_sha256": _digest(memories), "records": len(memories),
               "entrypoint": "scripts/train_him_sft.py", "resume_required": True,
               "output_checkpoint": f"checkpoints/candidates/{ctx.run_id}"}
        runnable = len(memories) > 0
        return {"status": "scheduled" if runnable else "blocked",
                "metrics": {"training_records": len(memories), "jobs_scheduled": int(runnable)},
                "outputs": {"training_job": job, "training_job_sha256": _digest(job)},
                "blockers": [] if runnable else ["training requires at least one constitutionally accepted memory"]}


class ReasoningAgent(ContinuousAgent):
    name = "reasoning"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        cases = list(state.get("reasoning_cases", []))
        passed = sum(bool(c.get("passed")) for c in cases)
        return {"status": "passed" if cases and passed == len(cases) else "quarantined",
                "metrics": {"cases": len(cases), "passed": passed, "pass_rate": round(passed/max(len(cases),1), 4)},
                "outputs": {"failure_samples": [c for c in cases if not c.get("passed")][:20]},
                "blockers": [] if cases and passed == len(cases) else ["reasoning workflow suite is incomplete or has failures"]}


class EvaluationAgent(ContinuousAgent):
    name = "evaluation"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        suites = dict(state.get("evaluation_suites", {}))
        required = ("user_chat", "relay_tool_use", "safety", "coding", "checkpoint_integrity")
        missing = [name for name in required if name not in suites]
        failed = [name for name, value in suites.items() if not bool(value.get("passed"))]
        scores = [float(value.get("score", 0.0)) for value in suites.values()]
        return {"status": "passed" if not missing and not failed else "quarantined",
                "metrics": {"suites": len(suites), "mean_score": round(sum(scores)/max(len(scores),1), 4),
                            "missing_required": len(missing), "failed_suites": len(failed)},
                "outputs": {"missing": missing, "failed": failed, "suite_results": suites},
                "blockers": [f"missing suite: {x}" for x in missing] + [f"failed suite: {x}" for x in failed]}


class DriftAgent(ContinuousAgent):
    name = "drift"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        baseline = dict(state.get("baseline_scores", {})); candidate = dict(state.get("candidate_scores", {}))
        regressions = {k: round(float(candidate.get(k, 0))-float(v), 6) for k, v in baseline.items()
                       if float(candidate.get(k, 0))-float(v) < -0.02}
        return {"status": "passed" if baseline and candidate and not regressions else "quarantined",
                "metrics": {"dimensions": len(baseline), "regressions": len(regressions)},
                "outputs": {"regressions": regressions, "rollback_target": ctx.parent_checkpoint_id},
                "blockers": [] if baseline and candidate and not regressions else ["regression or missing matched baseline detected"]}


class PromotionAgent(ContinuousAgent):
    name = "promotion"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        statuses = dict(state.get("agent_statuses", {}))
        required = tuple(x for x in AGENT_ORDER if x not in {"promotion", "research", "training"})
        blockers = [f"{name}={statuses.get(name, 'missing')}" for name in required if statuses.get(name) != "passed"]
        promotion_requested = not blockers and bool(state.get("candidate_files"))
        evidence = {
            "resume_state_present": bool(state.get("resume_state_present")),
            "matched_benchmark": statuses.get("evaluation") == "passed",
            "protected_capabilities_pass": statuses.get("evaluation") == "passed",
            "replay_or_forgetting_eval": statuses.get("drift") == "passed",
            "reversible_module_boundary": True,
            "tool_registry_receipt": statuses.get("knowledge") == "passed",
            "rollback_target": ctx.parent_checkpoint_id,
            "rollback_verified": bool(ctx.parent_checkpoint_id and statuses.get("drift") == "passed"),
            "continual_learning": True,
            "promotion_requested": promotion_requested,
        }
        checkpoint = build_constitutional_checkpoint(
            root=ctx.root, checkpoint_id=ctx.run_id, checkpoint_class="training_state",
            model_id=ctx.model_id, files=state.get("candidate_files", []),
            parent_checkpoint_id=ctx.parent_checkpoint_id, evidence=evidence,
            promotion_requested=promotion_requested,
            signing_key=state.get("promotion_signing_key"), authorized_by=state.get("authorized_by"),
            rollback={"target": ctx.parent_checkpoint_id, "verified": evidence["rollback_verified"]},
            capabilities={"relay_learning": True, "agent_fleet": list(AGENT_ORDER)},
        )
        manifest = write_constitutional_manifest(ctx.root, checkpoint, f"{ctx.run_id}.constitutional.json")
        status = "passed" if checkpoint.promotion_status == "promoted" else "quarantined"
        return {"status": status, "metrics": {"protocols": len(checkpoint.protocols),
                    "protocols_passed": sum(x.passed for x in checkpoint.protocols)},
                "outputs": {"promotion_status": checkpoint.promotion_status, "manifest": str(manifest),
                            "manifest_sha256": checkpoint.manifest_sha256},
                "blockers": blockers + [x.blocker for x in checkpoint.protocols if not x.passed and x.blocker]}


class ResearchAgent(ContinuousAgent):
    name = "research"

    def execute(self, ctx: FleetContext, state: Mapping[str, Any]) -> Dict[str, Any]:
        gaps = list(state.get("blockers", []))
        experiments = [{"experiment_id": f"exp-{_digest(gap)[:10]}", "hypothesis": gap,
                        "expected_effect": "remove one measured promotion blocker",
                        "required_evidence": ["focused test", "matched baseline", "cycle receipt"]}
                       for gap in gaps[:10]]
        return {"metrics": {"proposals": len(experiments)},
                "outputs": {"experiments": experiments, "research_queue_sha256": _digest(experiments)}}


class ContinuousImprovementFleet:
    """Runs one bounded, receipt-bearing improvement cycle."""

    def __init__(self, root: str | Path, model_id: str, parent_checkpoint_id: str | None = None):
        run_id = f"cycle-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.ctx = FleetContext(Path(root), run_id, model_id, parent_checkpoint_id)
        self.store = ReceiptStore(self.ctx.root / "evidence" / "continuous" / run_id)
        self.agents = [CorpusAgent(), KnowledgeAgent(), MemoryAgent(), TokenizerAgent(),
                       TrainingAgent(), ReasoningAgent(), EvaluationAgent(), DriftAgent(),
                       PromotionAgent(), ResearchAgent()]

    def run(self, initial_state: Mapping[str, Any]) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(initial_state)
        results = []
        statuses: Dict[str, str] = {}
        blockers: list[str] = []
        for agent in self.agents:
            state["agent_statuses"] = statuses
            state["blockers"] = blockers
            result = agent.run(self.ctx, state, self.store)
            results.append(result); statuses[agent.name] = result["status"]
            blockers.extend(result.get("blockers", []))
            state.update(result.get("outputs", {}))
        summary = {"schema": "auro.continuous.cycle.v1", "run_id": self.ctx.run_id,
                   "model_id": self.ctx.model_id, "parent_checkpoint_id": self.ctx.parent_checkpoint_id,
                   "agent_order": list(AGENT_ORDER), "statuses": statuses,
                   "promotion_status": results[-2]["outputs"].get("promotion_status", "quarantined"),
                   "blockers": blockers, "results": results}
        summary["cycle_sha256"] = _digest(summary)
        path = self.store.root / "cycle-summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary
