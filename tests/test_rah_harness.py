from auro_native_llm.rah import SCHEMA, PROTOCOL, _role_for, plan_fanout, run_rah
from auro_native_llm.types import SubAgentRole


def test_rah_schema_and_roles():
    assert SCHEMA == "auro.rah.v1"
    assert PROTOCOL == "AURO-RAH/1.0"
    assert _role_for("implement the parser") == SubAgentRole.CODE_EDIT
    assert _role_for("match two PSDs") == SubAgentRole.SPECTRAL_MATCH
    assert _role_for("plan the fan-out") == SubAgentRole.PLAN


def test_plan_fanout_splits_bullets():
    p = plan_fanout("- embed the corpus\n- match two PSDs\n- critique the train plan")
    assert len(p["leaves"]) == 3
    assert p["leaves"][0]["role"] == SubAgentRole.SPECTRAL_MATCH.value


def test_run_rah_parallel_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr("auro_native_llm.rah.ROOT", tmp_path)
    out = run_rah(
        "split the work",
        leaves=["embed the library", "match two records", "plan next train"],
        max_parallel=3,
        parent_model_id="Auro-14B",
    )
    assert out["leaves"] == 3
    assert out["parallel"] == 3
    assert (tmp_path / out["run_id"] / "RUN.json").is_file()
    assert "Auro RAH" in (out.get("synthesis") or "")
