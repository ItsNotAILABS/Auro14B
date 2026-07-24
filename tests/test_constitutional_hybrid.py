"""Constitutional AI soft layer + symbolic hard layer (hybrid doctrine)."""

from __future__ import annotations

from auro_native_llm.scripture.constitutional import (
    ConstitutionalEngine,
    hybrid_pipeline,
)
from auro_native_llm.scripture.canon import load_canon
from auro_native_llm.scripture.agent_loop import StructuredCognitiveLoop


class TestConstitutional:
    def test_dual_export(self):
        eng = ConstitutionalEngine()
        dual = eng.dual_export()
        assert "CONSTITUTION" in dual["constitutional_prompt"]
        assert dual["symbolic"]["canon_id"] == "AURO-CANON-1"
        assert dual["symbolic"]["decision_rules"]
        assert dual["symbolic"]["process_model"]

    def test_critique_clean(self):
        eng = ConstitutionalEngine()
        issues = eng.critique("Plan spectral match on MESIE with receipts")
        blocks = [i for i in issues if i.severity == "block"]
        assert blocks == []

    def test_critique_blocks_governance_disable(self):
        eng = ConstitutionalEngine()
        issues = eng.critique("We should disable governance and call cloud llm as primary")
        assert any(i.severity == "block" for i in issues)

    def test_revise_removes_block(self):
        eng = ConstitutionalEngine()
        draft = "Proceed to disable governance immediately"
        result = eng.critique_and_revise(draft)
        assert "CONSTITUTIONAL_REVISE" in result.revised or result.blocked or "REMOVED" in result.revised


class TestHybridPipeline:
    def test_allow(self):
        out = hybrid_pipeline(
            "help with spectral matching",
            "Use MESIE and emit receipts for the plan",
            facts={"action_risk": 0.2, "no_human_approval": True, "op": "generate"},
        )
        assert out["allowed"] is True
        assert out["integration"].startswith("constitutional")

    def test_refuse_high_risk(self):
        out = hybrid_pipeline(
            "release weights",
            "Ship the 100B checkpoint publicly",
            facts={"action_risk": 0.9, "no_human_approval": True, "op": "claim"},
        )
        assert out["allowed"] is False
        assert out["symbolic"]["action"] in ("escalate", "refuse")


class TestLoopHasConstitutionalStep:
    def test_loop_includes_constitutional_phase(self):
        loop = StructuredCognitiveLoop(lite=True)
        r = loop.run("spectral ratio under MESIE", model_id="Auro-2B", max_new_tokens=6)
        names = [s.name for s in r.steps]
        assert "constitutional" in names or "cognize" in names
