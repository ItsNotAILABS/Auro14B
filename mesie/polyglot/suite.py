"""AISVectorPolyglot — unified test, use, and integration suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from mesie.integration.ai_connector import AISystemConnector
from mesie.io.loaders import RecordInput
from mesie.polyglot.adapters import (
    JuliaAdapter,
    MotokoAdapter,
    PythonAdapter,
    RustAdapter,
    TypeScriptAdapter,
)
from mesie.polyglot.adapters.base import PolyglotAdapter
from mesie.polyglot.contract import (
    AISVectorMessage,
    AISVectorResponse,
    CONTRACT_VERSION,
    PolyglotAction,
    RuntimeId,
    SUITE_NAME,
    record_to_dict,
)
from mesie.polyglot.third_party_ai import ThirdPartyAIConnector
from mesie.polyglot.vector_bridge import AISVectorBridge, VectorQueryResult


DEFAULT_ROUTING: Dict[PolyglotAction, RuntimeId] = {
    PolyglotAction.VALIDATE: RuntimeId.MOTOKO,
    PolyglotAction.MATCH: RuntimeId.RUST,
    PolyglotAction.EMBED: RuntimeId.PYTHON,
    PolyglotAction.RANK: RuntimeId.PYTHON,
    PolyglotAction.FINGERPRINT: RuntimeId.PYTHON,
    PolyglotAction.HEALTH: RuntimeId.PYTHON,
}


@dataclass
class SuiteHealth:
    name: str = SUITE_NAME
    contract_version: str = CONTRACT_VERSION
    runtimes: Dict[str, dict] = field(default_factory=dict)
    vector_indexed: int = 0
    ais_connected: bool = True


class AISVectorPolyglotSuite:
    """Named polyglot engine: Julia, Python, Motoko, Rust, TypeScript + vector + AIS + 3rd-party AI."""

    def __init__(
        self,
        routing: Optional[Dict[PolyglotAction, RuntimeId]] = None,
        motoko_url: str = "",
    ) -> None:
        self.routing = dict(DEFAULT_ROUTING)
        if routing:
            self.routing.update(routing)
        self._adapters: Dict[RuntimeId, PolyglotAdapter] = {
            RuntimeId.PYTHON: PythonAdapter(),
            RuntimeId.RUST: RustAdapter(),
            RuntimeId.JULIA: JuliaAdapter(),
            RuntimeId.MOTOKO: MotokoAdapter(canister_url=motoko_url),
            RuntimeId.TYPESCRIPT: TypeScriptAdapter(),
        }
        self.vector = AISVectorBridge()
        self.ais = AISystemConnector()
        self.third_party = ThirdPartyAIConnector(self.dispatch)

    def adapter(self, runtime: RuntimeId) -> PolyglotAdapter:
        return self._adapters[runtime]

    def dispatch(self, message: AISVectorMessage) -> AISVectorResponse:
        runtime = message.runtime or self.routing.get(message.action, RuntimeId.PYTHON)
        message.runtime = runtime
        return self._adapters[runtime].dispatch(message)

    def validate(self, record: RecordInput, runtime: Optional[RuntimeId] = None) -> AISVectorResponse:
        return self.dispatch(
            AISVectorMessage(
                action=PolyglotAction.VALIDATE,
                runtime=runtime or self.routing[PolyglotAction.VALIDATE],
                record=record_to_dict(record),
            )
        )

    def match(
        self,
        reference: RecordInput,
        candidate: RecordInput,
        runtime: Optional[RuntimeId] = None,
    ) -> AISVectorResponse:
        return self.dispatch(
            AISVectorMessage(
                action=PolyglotAction.MATCH,
                runtime=runtime or self.routing[PolyglotAction.MATCH],
                record_a=record_to_dict(reference),
                record_b=record_to_dict(candidate),
            )
        )

    def embed(self, record: RecordInput, runtime: Optional[RuntimeId] = None) -> AISVectorResponse:
        return self.dispatch(
            AISVectorMessage(
                action=PolyglotAction.EMBED,
                runtime=runtime or self.routing[PolyglotAction.EMBED],
                record=record_to_dict(record),
            )
        )

    def index_corpus(self, records: Sequence[RecordInput]) -> int:
        return self.vector.index(records)

    def vector_query(self, record: RecordInput, top_k: int = 5) -> VectorQueryResult:
        return self.vector.query(record, top_k=top_k)

    def ais_process(self, record: RecordInput) -> Dict[str, Any]:
        return self.ais.ingest(record)

    def health(self) -> SuiteHealth:
        return SuiteHealth(
            runtimes={rid.value: ad.health() for rid, ad in self._adapters.items()},
            vector_indexed=self.vector.state.n_indexed,
        )

    def parity_matrix(
        self,
        reference: RecordInput,
        candidate: RecordInput,
    ) -> Dict[str, Any]:
        """Run match across all runtimes for integration testing."""
        results = {}
        for rid in (RuntimeId.PYTHON, RuntimeId.RUST, RuntimeId.JULIA, RuntimeId.MOTOKO, RuntimeId.TYPESCRIPT):
            resp = self.match(reference, candidate, runtime=rid)
            results[rid.value] = {
                "ok": resp.ok,
                "score": resp.data.get("composite_score"),
                "mode": self._adapters[rid].mode,
                "latency_ms": resp.latency_ms,
            }
        scores = [r["score"] for r in results.values() if r["ok"] and r["score"] is not None]
        spread = max(scores) - min(scores) if scores else None
        return {"results": results, "score_spread": spread, "parity_ok": spread is not None and spread < 0.25}