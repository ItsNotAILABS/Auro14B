"""Scriptural runtime substrate — live binding of canon to generate/train/dispatch.

This is the running-time substrate: every LLM act goes through executor +
memory + governance. Symbols construct behavior.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from auro_native_llm.scripture.canon import Canon, load_canon
from auro_native_llm.scripture.executor import ExecutionVerdict, Operation, ScripturalExecutor
from auro_native_llm.scripture.governance import InnerGovernance
from auro_native_llm.scripture.memory import ScripturalMemory


@dataclass
class SubstrateResult:
    ok: bool
    op: str
    verdict: Dict[str, Any]
    output: Any = None
    memory_context: str = ""
    refusal: Optional[str] = None
    latency_ms: float = 0.0
    scriptural: bool = True
    compute_plane: str = "MESIE"

    def to_dict(self) -> Dict[str, Any]:
        out = self.output
        if hasattr(out, "to_dict"):
            out = out.to_dict()
        return {
            "schema": "auro.scripture.substrate_result.v1",
            "ok": self.ok,
            "op": self.op,
            "verdict": self.verdict,
            "output": out,
            "memory_context": self.memory_context,
            "refusal": self.refusal,
            "latency_ms": self.latency_ms,
            "scriptural": True,
            "compute_plane": self.compute_plane,
        }


class ScripturalSubstrate:
    """Doctrine-bound runtime for Auro LLMs + multi-embedded agents."""

    def __init__(
        self,
        canon: Optional[Canon] = None,
        canon_path: Optional[str] = None,
        memory: Optional[ScripturalMemory] = None,
        memory_path: Optional[str] = None,
        *,
        lite: bool = True,
        model: Any = None,
    ) -> None:
        self.canon = canon or load_canon(canon_path)
        mem_cfg = self.canon.memory or {}
        if memory is not None:
            self.memory = memory
        elif memory_path and Path(memory_path).exists():
            self.memory = ScripturalMemory.load(memory_path)
        else:
            self.memory = ScripturalMemory(
                capacity=int(mem_cfg.get("capacity", 2048)),
                embed_dim=int(mem_cfg.get("embed_dim", 256)),
                decay=float(mem_cfg.get("decay", 0.98)),
                require_canon_tag=bool(mem_cfg.get("require_canon_tag", True)),
            )
        self.executor = ScripturalExecutor(self.canon)
        self.governance = InnerGovernance(self.canon)
        self._lm = model  # optional injected AuroLanguageModel
        self.lite = lite
        self.memory_path = memory_path

    def health(self) -> Dict[str, Any]:
        return {
            "substrate": "ScripturalSubstrate",
            "canon_id": self.canon.canon_id,
            "canon_sha256": self.canon.content_sha256,
            "compute_plane": self.canon.compute_plane,
            "principle": self.canon.principle,
            "memory": self.memory.stats(),
            "receipts": len(self.executor.history),
            "prior_receipt": self.executor.prior_receipt_hash,
            "fail_closed": self.canon.governance.get("fail_closed", True),
            "scriptural": True,
        }

    def _get_lm(self, model_id: str = "Auro-2B"):
        if self._lm is not None and getattr(self._lm, "model_id", None) == model_id:
            return self._lm
        from auro_native_llm.model.auro_lm import AuroLanguageModel

        if self.lite:
            # Executable inner-loop model for governance substrate (still real MESIE SpectralGPT)
            self._lm = AuroLanguageModel.build(
                model_id,
                mode="dev",
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                head_dim=16,
                ffn_dim=128,
                num_experts=2,
                top_k_experts=1,
                max_seq_len=128,
                vocab_size=512,
                use_moe=True,
            )
        else:
            self._lm = AuroLanguageModel.build(model_id, mode="dev")
        return self._lm

    def generate(
        self,
        prompt: str,
        *,
        model_id: str = "Auro-2B",
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.85,
        use_memory: bool = True,
    ) -> SubstrateResult:
        t0 = time.perf_counter()
        verdict = self.executor.execute(
            Operation.GENERATE,
            intent=prompt,
            model_id=model_id,
        )
        if not verdict.allowed:
            refusal = self.governance.refusal_text(
                self.governance.review("generate", prompt, model_id=model_id),
                prompt,
            )
            # still write refusal to memory — law constructs world
            self.memory.write(
                f"REFUSAL: {prompt[:200]}",
                canon_id=self.canon.canon_id,
                model_id=model_id,
                op="generate",
                article_ids=verdict.articles_applied,
                importance=0.9,
                metadata={"execution_id": verdict.execution_id, "refused": True},
            )
            return SubstrateResult(
                ok=False,
                op="generate",
                verdict=verdict.to_dict(),
                refusal=refusal,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        mem_block = self.memory.context_block(prompt, top_k=3) if use_memory else ""
        preamble = verdict.governance.get("preamble") or ""
        full_prompt = "\n\n".join(x for x in (preamble, mem_block, prompt) if x)

        max_tok = max_new_tokens or int(
            self.canon.governance.get("max_generate_tokens_default", 128)
        )
        lm = self._get_lm(model_id)
        gen = lm.generate(full_prompt, max_new_tokens=max_tok, temperature=temperature)

        # Embed memory of the act
        self.memory.write(
            f"GEN:{prompt[:120]} → {gen.text[:200]}",
            canon_id=self.canon.canon_id,
            model_id=model_id,
            op="generate",
            article_ids=verdict.articles_applied,
            importance=0.7,
            embedding=None,
            metadata={
                "execution_id": verdict.execution_id,
                "receipt": verdict.receipt_hash,
                "num_params": gen.num_params,
            },
        )
        # attach scriptural meta onto generation dict
        payload = gen.to_dict()
        payload["scriptural"] = True
        payload["canon_id"] = self.canon.canon_id
        payload["execution_id"] = verdict.execution_id
        payload["receipt_hash"] = verdict.receipt_hash

        return SubstrateResult(
            ok=True,
            op="generate",
            verdict=verdict.to_dict(),
            output=payload,
            memory_context=mem_block,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def train_step_governed(
        self,
        model_id: str,
        batch_text: str,
        *,
        claims_trained_checkpoint: bool = False,
    ) -> SubstrateResult:
        t0 = time.perf_counter()
        verdict = self.executor.execute(
            Operation.TRAIN,
            intent=f"train_step:{batch_text[:200]}",
            model_id=model_id,
            claims_trained_checkpoint=claims_trained_checkpoint,
        )
        if not verdict.allowed:
            return SubstrateResult(
                ok=False,
                op="train",
                verdict=verdict.to_dict(),
                refusal=verdict.message,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        lm = self._get_lm(model_id)
        import numpy as np

        ids = np.array(
            [lm.tokenizer.encode(batch_text, max_length=min(96, lm.config.max_seq_len))],
            dtype=np.int64,
        )
        metrics = lm.train_step(ids, ids, text_for_meaning=batch_text)
        self.memory.write(
            f"TRAIN:{batch_text[:160]} loss={metrics.get('loss', 0):.4f}",
            canon_id=self.canon.canon_id,
            model_id=model_id,
            op="train",
            article_ids=verdict.articles_applied,
            importance=0.8,
            metadata={"metrics": metrics, "execution_id": verdict.execution_id},
        )
        return SubstrateResult(
            ok=True,
            op="train",
            verdict=verdict.to_dict(),
            output={"metrics": metrics, "model_id": model_id, "train_steps": lm.train_steps},
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def dispatch(
        self,
        role: str,
        intent: str,
        *,
        parent_model_id: str = "Auro-14B",
    ) -> SubstrateResult:
        t0 = time.perf_counter()
        # Resolve child via family router for host check
        child_id = ""
        try:
            from auro_native_llm.subagents import MultiEmbeddedSubAgentRouter
            from auro_native_llm.types import SubAgentRole

            router = MultiEmbeddedSubAgentRouter(parent_model_id=parent_model_id)
            try:
                role_enum: Any = SubAgentRole(role)
            except ValueError:
                role_enum = role
            route = router.dispatch(role_enum, intent)
            child_id = route.child_model_id if route.ok else ""
            host_ok = bool(route.ok)
        except Exception:
            host_ok = False
            child_id = ""

        verdict = self.executor.execute(
            Operation.DISPATCH,
            intent=intent,
            parent_model_id=parent_model_id,
            child_model_id=child_id or "unknown",
            model_id=parent_model_id,
            host_allowed=host_ok,
            metadata={"role": role},
        )
        if not verdict.allowed or not host_ok:
            return SubstrateResult(
                ok=False,
                op="dispatch",
                verdict=verdict.to_dict(),
                refusal=verdict.message if not verdict.allowed else "host matrix / route failed",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Run child generate under scripture
        child_result = self.generate(intent, model_id=child_id, max_new_tokens=64)
        self.memory.write(
            f"DISPATCH:{parent_model_id}->{child_id}/{role}: {intent[:120]}",
            canon_id=self.canon.canon_id,
            model_id=parent_model_id,
            op="dispatch",
            article_ids=verdict.articles_applied,
            importance=0.85,
            metadata={"child": child_id, "role": role, "execution_id": verdict.execution_id},
        )
        return SubstrateResult(
            ok=child_result.ok,
            op="dispatch",
            verdict=verdict.to_dict(),
            output={
                "parent": parent_model_id,
                "child": child_id,
                "role": role,
                "child_generation": child_result.to_dict(),
            },
            memory_context=child_result.memory_context,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def claim(
        self,
        statement: str,
        *,
        model_id: str,
        claims_trained_checkpoint: bool = False,
        has_checkpoint_receipt: bool = False,
        has_eval_receipt: bool = False,
    ) -> SubstrateResult:
        t0 = time.perf_counter()
        verdict = self.executor.execute(
            Operation.CLAIM,
            intent=statement,
            model_id=model_id,
            claims_trained_checkpoint=claims_trained_checkpoint,
            has_checkpoint_receipt=has_checkpoint_receipt,
            has_eval_receipt=has_eval_receipt,
        )
        return SubstrateResult(
            ok=verdict.allowed,
            op="claim",
            verdict=verdict.to_dict(),
            output={"statement": statement, "accepted": verdict.allowed},
            refusal=None if verdict.allowed else verdict.message,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def persist_memory(self, path: Optional[str] = None) -> str:
        p = path or self.memory_path or "deliverables/auro_scripture/memory.json"
        self.memory.save(p)
        return p

    def save_receipts(self, path: str = "deliverables/auro_scripture/receipts.jsonl") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for v in self.executor.history:
                f.write(json.dumps(v.to_dict()) + "\n")
        return path
