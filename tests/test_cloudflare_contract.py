from auro_native_llm.cloudflare import CloudflareRuntimeContract


def test_cloudflare_plane_is_opt_in_and_search_then_execute(monkeypatch) -> None:
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    contract=CloudflareRuntimeContract(); manifest=contract.manifest(); recipe=contract.recipe("create a Worker")
    assert manifest["enabled_by_default"] is False
    assert manifest["configured"] is False and manifest["credential_exposed"] is False
    assert [step["tool"] for step in recipe["steps"]]==["search","execute"]
    assert recipe["steps"][1]["approval_required"] is True and recipe["executed"] is False
