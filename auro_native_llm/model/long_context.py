"""Hierarchical long-context surface for AuroLanguageModel.

AURO can accept up to 294,912 tokens while keeping dense MESIE attention
bounded. History is no longer selected only by token-overlap heuristics: it is
retrieved coarse-to-fine in AURO's learned embedding space at macro, meso, and
micro scales. Retrieved historical chunks are summarized into hidden space and
prime active working memory before the transformer pass, so long-range memory
can influence adaptive MoE compute on the same cycle.

Accepted context is still distinct from simultaneously dense-attended context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from auro_native_llm.context import ContextEnvelope
from auro_native_llm.model.auro_lm import AuroLanguageModel
from auro_native_llm.model.hierarchical_context import (
    HierarchicalContextMemory,
    HierarchicalContextReceipt,
)


@dataclass
class LongContextForward:
    outputs: Dict[str, Any]
    receipt: HierarchicalContextReceipt


class AuroLongContextModel:
    """Model-aware hierarchical long-context surface over executable AURO."""

    accepted_context_tokens = 294_912

    def __init__(self, model: AuroLanguageModel, *, dense_window: Optional[int] = None) -> None:
        self.model = model
        dense = min(int(dense_window or model.config.max_seq_len), 32_768)
        # Preserve the established geometry object for compatibility while the
        # hierarchy performs the actual history selection.
        self.envelope = ContextEnvelope(
            accepted_limit=self.accepted_context_tokens,
            dense_window=dense,
            chunk_size=min(4096, max(256, dense // 4)),
            retrieval_budget=min(8192, dense // 3),
        )
        micro = min(512, max(128, self.envelope.retrieval_budget // 8 or 128))
        meso = max(micro * 4, 1024)
        macro = max(meso * 4, 4096)
        self.hierarchy = HierarchicalContextMemory(
            model,
            envelope=self.envelope,
            micro_size=micro,
            meso_size=meso,
            macro_size=macro,
        )
        self.last_context_receipt: Optional[HierarchicalContextReceipt] = None

    @property
    def config(self):
        return self.model.config

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def prepare_context(self, token_ids: Sequence[int] | np.ndarray):
        dense, receipt, chunks = self.hierarchy.ingest(token_ids)
        self.last_context_receipt = receipt
        return dense, receipt, chunks

    def forward_ids(
        self,
        token_ids,
        *,
        text_for_meaning: Optional[str] = None,
        spectral_record: Any = None,
    ) -> LongContextForward:
        dense, receipt, _ = self.prepare_context(token_ids)

        # Only true retrieved history primes memory. Short prompts remain on the
        # normal model path and are not represented twice.
        primed = False
        if receipt.retrieved_tokens > 0:
            primed = self.hierarchy.prime_working_memory()
            if primed and self.hierarchy.last_receipt is not None:
                receipt = self.hierarchy.last_receipt
                self.last_context_receipt = receipt

        outputs = self.model.forward_ids(
            dense,
            text_for_meaning=text_for_meaning,
            spectral_record=spectral_record,
        )
        outputs["context_receipt"] = receipt.to_dict()
        outputs["accepted_context_tokens"] = self.accepted_context_tokens
        outputs["simultaneously_attended_tokens"] = int(dense.size)
        outputs["hierarchical_context"] = {
            "schema": "auro.hierarchical-context-runtime.v1",
            "macro_candidates": receipt.macro_candidates,
            "meso_candidates": receipt.meso_candidates,
            "micro_candidates": receipt.micro_candidates,
            "selected_micro_ids": list(receipt.selected_micro_ids),
            "retrieved_tokens": receipt.retrieved_tokens,
            "working_memory_query_used": receipt.working_memory_query_used,
            "working_memory_primed": bool(primed),
            "summary_norm": receipt.summary_norm,
        }
        return LongContextForward(outputs=outputs, receipt=receipt)

    def info(self) -> Dict[str, Any]:
        info = self.model.info()
        info["context"] = {
            "schema": "auro.context.capability.v2",
            "accepted_context_tokens": self.accepted_context_tokens,
            "dense_attention_tokens": self.envelope.dense_window,
            "retrieval_budget_tokens": self.envelope.retrieval_budget,
            "architecture": (
                "learned-embedding hierarchical macro/meso/micro retrieval -> "
                "historical hidden summary -> active working-memory priming -> "
                "bounded dense MESIE attention"
            ),
            "working_memory_query": True,
            "same_cycle_compute_priming": True,
            "claim_boundary": "accepted context is not simultaneous dense attention",
        }
        return info
