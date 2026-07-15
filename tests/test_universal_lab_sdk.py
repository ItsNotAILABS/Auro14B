"""Tests for the UniversalLabSDK."""

from mesie.sdk.universal_lab_sdk import UniversalLabSDK


class TestUniversalLabSDK:
    def test_init(self):
        sdk = UniversalLabSDK()
        info = sdk.info()
        assert "spectral" in info["labs"]
        assert "arxiv" in info["data_sources"]

    def test_list_labs(self):
        sdk = UniversalLabSDK()
        labs = sdk.list_labs()
        assert len(labs) == 5

    def test_lab_access(self):
        sdk = UniversalLabSDK()
        physics = sdk.lab("physics")
        assert physics.domain == "physics"

    def test_lab_not_found(self):
        sdk = UniversalLabSDK()
        try:
            sdk.lab("nonexistent")
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_run_lab(self):
        sdk = UniversalLabSDK()
        result = sdk.run_lab("chemistry", "formula_parse", formula="CO2")
        assert result.status == "success"
        assert result.data["elements"]["C"] == 1
        assert result.data["elements"]["O"] == 2

    def test_research(self):
        sdk = UniversalLabSDK()
        result = sdk.research("What are the properties of water?")
        assert result.status in ("completed", "failed")
        assert result.report is not None

    def test_thesis(self):
        sdk = UniversalLabSDK()
        result = sdk.thesis("Water Study", "Water boils at 100C at sea level")
        assert result.thesis_result is not None

    def test_data_source_access(self):
        sdk = UniversalLabSDK()
        arxiv = sdk.data_source("arxiv")
        assert arxiv.name == "arxiv"

    def test_data_source_not_found(self):
        sdk = UniversalLabSDK()
        try:
            sdk.data_source("nonexistent")
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_hub_access(self):
        sdk = UniversalLabSDK()
        assert sdk.hub is not None
        info = sdk.hub.info()
        assert "labs" in info
