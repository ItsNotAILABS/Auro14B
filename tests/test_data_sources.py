"""Tests for data sources module."""

from mesie.data_sources.cache import DataCache, CacheConfig
from mesie.data_sources.arxiv import ArxivSource
from mesie.data_sources.pubchem import PubChemSource
from mesie.data_sources.usgs import USGSSource
from mesie.data_sources.registry import DataSourceRegistry, build_default_sources


class TestDataCache:
    def test_put_and_get(self):
        cache = DataCache()
        cache.put("key1", {"value": 42})
        assert cache.get("key1") == {"value": 42}

    def test_missing_key(self):
        cache = DataCache()
        assert cache.get("nonexistent") is None

    def test_invalidate(self):
        cache = DataCache()
        cache.put("key1", {"x": 1})
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = DataCache()
        cache.put("a", {})
        cache.put("b", {})
        cache.clear()
        assert cache.size == 0


class TestArxivSource:
    def test_search(self):
        source = ArxivSource()
        result = source.search("spectral analysis")
        assert result["source"] == "arxiv"
        assert result["query"] == "spectral analysis"

    def test_fetch_paper(self):
        source = ArxivSource()
        result = source.fetch_paper("2301.12345")
        assert result["arxiv_id"] == "2301.12345"
        assert "arxiv.org" in result["url"]

    def test_categories(self):
        source = ArxivSource()
        cats = source.categories()
        assert "cs.AI" in cats

    def test_caching(self):
        source = ArxivSource()
        result1 = source.search("test query")
        result2 = source.search("test query")
        assert result1 == result2


class TestPubChemSource:
    def test_search_compound(self):
        source = PubChemSource()
        result = source.search_compound("aspirin")
        assert result["source"] == "pubchem"
        assert result["query"] == "aspirin"

    def test_get_compound(self):
        source = PubChemSource()
        result = source.get_compound(2244)
        assert result["cid"] == 2244

    def test_get_structure(self):
        source = PubChemSource()
        result = source.get_structure(2244, format="smiles")
        assert result["format"] == "smiles"


class TestUSGSSource:
    def test_search_earthquakes(self):
        source = USGSSource()
        result = source.search_earthquakes(min_magnitude=5.0)
        assert result["source"] == "usgs_earthquake"

    def test_get_earthquake(self):
        source = USGSSource()
        result = source.get_earthquake("us7000test")
        assert result["event_id"] == "us7000test"

    def test_available_datasets(self):
        source = USGSSource()
        datasets = source.available_datasets()
        assert len(datasets) >= 3
        ids = [d["id"] for d in datasets]
        assert "earthquake" in ids


class TestDataSourceRegistry:
    def test_build_default(self):
        registry = build_default_sources()
        assert "arxiv" in registry.names
        assert "pubchem" in registry.names
        assert "usgs" in registry.names

    def test_list_sources(self):
        registry = build_default_sources()
        sources = registry.list_sources()
        assert len(sources) == 3
        assert all("name" in s and "description" in s for s in sources)
