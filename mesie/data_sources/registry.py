"""Data source registry — central discovery for all external data connectors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.data_sources.cache import CacheConfig, DataCache


class DataSourceRegistry:
    """Registry of available external data sources.

    Provides discovery and unified access to all data source connectors.
    """

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache()
        self._sources: Dict[str, Any] = {}

    def register(self, source: Any) -> None:
        """Register a data source by its name attribute."""
        self._sources[source.name] = source

    def get(self, name: str) -> Optional[Any]:
        """Get a data source by name."""
        return self._sources.get(name)

    def list_sources(self) -> List[Dict[str, str]]:
        """List all registered sources."""
        return [
            {"name": s.name, "description": s.description}
            for s in self._sources.values()
        ]

    @property
    def names(self) -> List[str]:
        return list(self._sources.keys())

    @property
    def cache(self) -> DataCache:
        return self._cache


def build_default_sources(cache: Optional[DataCache] = None) -> DataSourceRegistry:
    """Create a registry with all built-in data sources."""
    from mesie.data_sources.arxiv import ArxivSource
    from mesie.data_sources.pubchem import PubChemSource
    from mesie.data_sources.usgs import USGSSource

    shared_cache = cache or DataCache()
    registry = DataSourceRegistry(cache=shared_cache)
    registry.register(ArxivSource(cache=shared_cache))
    registry.register(PubChemSource(cache=shared_cache))
    registry.register(USGSSource(cache=shared_cache))
    return registry
