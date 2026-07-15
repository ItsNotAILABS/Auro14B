"""Full test + use + integration suite for AISVectorPolyglot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

from mesie.polyglot.contract import PolyglotAction, RuntimeId, SUITE_NAME
from mesie.polyglot.suite import AISVectorPolyglotSuite


@dataclass
class IntegrationCase:
    id: str
    name: str
    category: str
    fn: Callable[[AISVectorPolyglotSuite], None]


@dataclass
class IntegrationOutcome:
    id: str
    name: str
    category: str
    passed: bool
    latency_ms: float
    error: str = ""


@dataclass
class IntegrationReport:
    suite: str
    passed: int
    failed: int
    total: int
    runtime_ms: float
    outcomes: List[IntegrationOutcome] = field(default_factory=list)


def _build_cases(refs: list) -> List[IntegrationCase]:
    ref, cand = refs[0], refs[1] if len(refs) > 1 else refs[0]

    def t01(s: AISVectorPolyglotSuite):
        r = s.health()
        assert r.name == SUITE_NAME
        assert len(r.runtimes) >= 5

    def t02(s: AISVectorPolyglotSuite):
        assert s.validate(ref, RuntimeId.PYTHON).ok

    def t03(s: AISVectorPolyglotSuite):
        assert s.validate(ref, RuntimeId.TYPESCRIPT).ok

    def t04(s: AISVectorPolyglotSuite):
        assert s.validate(ref, RuntimeId.MOTOKO).ok

    def t05(s: AISVectorPolyglotSuite):
        assert s.validate(ref, RuntimeId.RUST).ok

    def t06(s: AISVectorPolyglotSuite):
        assert s.validate(ref, RuntimeId.JULIA).ok

    def t07(s: AISVectorPolyglotSuite):
        m = s.match(ref, cand, RuntimeId.PYTHON)
        assert m.ok and m.data["composite_score"] >= 0

    def t08(s: AISVectorPolyglotSuite):
        m = s.match(ref, cand, RuntimeId.RUST)
        assert m.ok

    def t09(s: AISVectorPolyglotSuite):
        m = s.match(ref, cand, RuntimeId.TYPESCRIPT)
        assert m.ok

    def t10(s: AISVectorPolyglotSuite):
        p = s.parity_matrix(ref, cand)
        assert p["parity_ok"]

    def t11(s: AISVectorPolyglotSuite):
        e = s.embed(ref, RuntimeId.PYTHON)
        assert e.ok and e.vector and len(e.vector) >= 8

    def t12(s: AISVectorPolyglotSuite):
        n = s.index_corpus(refs)
        assert n == len(refs)

    def t13(s: AISVectorPolyglotSuite):
        q = s.vector_query(ref)
        assert q.neighbors and q.embedding

    def t14(s: AISVectorPolyglotSuite):
        q = s.vector_query(ref)
        assert q.fingerprint_hits or q.neighbors

    def t15(s: AISVectorPolyglotSuite):
        ais = s.ais_process(ref)
        assert "embedding" in ais or "record_id" in ais

    def t16(s: AISVectorPolyglotSuite):
        tools = s.third_party.openai_tools_schema()
        assert any(t["function"]["name"] == "mesie_match_spectra" for t in tools)

    def t17(s: AISVectorPolyglotSuite):
        out = s.third_party.invoke_tool("mesie_validate_spectrum", {"record": s.vector.export_contract_record(ref)})
        assert out.get("ok")

    def t18(s: AISVectorPolyglotSuite):
        out = s.third_party.invoke_tool("mesie_embed_spectrum", {"record": s.vector.export_contract_record(ref)})
        assert out.get("ok") and out.get("vector")

    def t19(s: AISVectorPolyglotSuite):
        from mesie.internal_api.bus import InternalBus
        from mesie.engines.polyglot_engine import PolyglotEngine

        bus = InternalBus()
        eng = PolyglotEngine(s)
        bus.register_engine(eng.name, eng.handle)
        from mesie.internal_api.messages import MessageEnvelope, MessageTopic

        msg = MessageEnvelope(
            topic=MessageTopic.ENGINE_REQUEST,
            source="test",
            target="polyglot",
            action="match",
            payload={"record_a": ref, "record_b": cand},
        )
        resp = eng.handle(msg)
        assert resp and resp.ok

    def t20(s: AISVectorPolyglotSuite):
        for action in PolyglotAction:
            if action == PolyglotAction.FINGERPRINT:
                continue
            rid = s.routing.get(action, RuntimeId.PYTHON)
            assert rid in s._adapters

    cases = [
        ("I01", "Suite health all runtimes", "core", t01),
        ("I02", "Python validate", "python", t02),
        ("I03", "TypeScript validate", "typescript", t03),
        ("I04", "Motoko validate mirror", "motoko", t04),
        ("I05", "Rust validate", "rust", t05),
        ("I06", "Julia validate", "julia", t06),
        ("I07", "Python match", "python", t07),
        ("I08", "Rust match", "rust", t08),
        ("I09", "TypeScript match", "typescript", t09),
        ("I10", "Cross-runtime parity matrix", "parity", t10),
        ("I11", "Python embed vector", "vector", t11),
        ("I12", "Vector bridge index corpus", "vector", t12),
        ("I13", "Vector ANN query", "vector", t13),
        ("I14", "Fingerprint hits", "vector", t14),
        ("I15", "AIS connector process", "ais", t15),
        ("I16", "Third-party OpenAI tools schema", "third_party", t16),
        ("I17", "Third-party validate tool invoke", "third_party", t17),
        ("I18", "Third-party embed tool invoke", "third_party", t18),
        ("I19", "Internal bus polyglot engine", "bus", t19),
        ("I20", "Routing table completeness", "core", t20),
    ]
    return [IntegrationCase(cid, name, cat, fn) for cid, name, cat, fn in cases]


def run_integration_suite(refs: list) -> IntegrationReport:
    suite = AISVectorPolyglotSuite()
    cases = _build_cases(refs)
    t0 = time.perf_counter()
    outcomes: List[IntegrationOutcome] = []
    for case in cases:
        t1 = time.perf_counter()
        try:
            case.fn(suite)
            outcomes.append(IntegrationOutcome(case.id, case.name, case.category, True, (time.perf_counter() - t1) * 1000))
        except Exception as exc:
            outcomes.append(
                IntegrationOutcome(
                    case.id, case.name, case.category, False, (time.perf_counter() - t1) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    passed = sum(1 for o in outcomes if o.passed)
    return IntegrationReport(
        suite=SUITE_NAME,
        passed=passed,
        failed=len(outcomes) - passed,
        total=len(outcomes),
        runtime_ms=(time.perf_counter() - t0) * 1000,
        outcomes=outcomes,
    )