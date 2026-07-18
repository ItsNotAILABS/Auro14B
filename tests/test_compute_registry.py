from auro_native_llm.production_fleet.compute import ComputeRegistry


def test_compute_registry_has_no_implicit_remote_fallback() -> None:
    manifest = ComputeRegistry('[{"id":"gpu-a","base_url":"https://compute.example/v1","model":"auro"}]').manifest()
    assert manifest["default"] == "embedded-browser"
    assert manifest["remote_fallback"] is False
    assert [item["plane"] for item in manifest["engines"]] == ["browser", "local", "cloud"]
