"""Near-300k context facade for AuroLanguageModel.

Accepts 294,912 tokens, reduces them through the governed ContextEnvelope, and
forwards only the bounded dense view into MESIE. Receipts preserve exactly what
was accepted, retrieved, truncated, and simultaneously attended.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from auro_native_llm.context import ContextEnvelope, ContextReceipt
from auro_native_llm.model.auro_lm import AuroLanguageModel


@dataclass
class LongContextForward:
    outputs: Dict[str, Any]
    receipt: ContextReceipt


class AuroLongContextModel:
    """Governed long-context surface over an executable AURO model."""

    accepted_context_tokens = 294_912

    def __init__(self, model: AuroLanguageModel, *, dense_window: Optional[int] = None) -> None:
        self.model = model
        dense = min(int(dense_window or model.config.max_seq_len), 32_768)
        self.envelope = ContextEnvelope(
            accepted_limit=self.accepted_context_tokens,
            dense_window=dense,
            chunk_size=min(4096, max(256, dense // 4)),
            retrieval_budget=min(8192, dense // 3),
        )
        self.last_context_receipt: Optional[ContextReceipt] = None

    @property
    def config(self):
        return self.model.config

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def prepare_context(self, token_ids: Sequence[int] | np.ndarray):
        dense, receipt, chunks = self.envelope.ingest(token_ids)
        self.last_context_receipt = receipt
        return dense, receipt, chunks

    def forward_ids(self, token_ids, *, text_for_meaning: Optional[str] = None, spectral_record: Any = None) -> LongContextForward:
        dense, receipt, _ = self.prepare_context(token_ids)
        outputs = self.model.forward_ids(dense, text_for_meaning=text_for_meaning, spectral_record=spectral_record)
        outputs["context_receipt"] = receipt.to_dict()
        outputs["accepted_context_tokens"] = self.accepted_context_tokens
        outputs["simultaneously_attended_tokens"] = int(dense.size)
        return LongContextForward(outputs=outputs, receipt=receipt)

    def info(self) -> Dict[str, Any]:
        info = self.model.info()
        info["context"] = {
            "schema": "auro.context.capability.v1",
            "accepted_context_tokens": self.accepted_context_tokens,
            "dense_attention_tokens": self.envelope.dense_window,
            "retrieval_budget_tokens": self.envelope.retrieval_budget,
            "architecture": "governed chunked context envelope with bounded dense MESIE attention",
            "claim_boundary": "accepted context is not simultaneous dense attention",
        }
        return info
