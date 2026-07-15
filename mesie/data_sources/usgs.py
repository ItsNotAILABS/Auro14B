"""USGS data source — earthquake catalog, geology, and earth science data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.data_sources.cache import DataCache


class USGSSource:
    """Connector to USGS APIs for earth science data.

    Provides access to earthquake catalogs, geological data, and
    earth observation datasets.

    Args:
        cache: Optional data cache instance.
    """

    EARTHQUAKE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    WATER_URL = "https://waterservices.usgs.gov/nwis"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache()

    @property
    def name(self) -> str:
        return "usgs"

    @property
    def description(self) -> str:
        return "USGS — earthquakes, geology, water, and earth observation data"

    def search_earthquakes(
        self,
        *,
        min_magnitude: float = 4.0,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_radius_km: float = 500.0,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search USGS earthquake catalog.

        Args:
            min_magnitude: Minimum earthquake magnitude.
            start_time: Start date (ISO format YYYY-MM-DD).
            end_time: End date (ISO format YYYY-MM-DD).
            latitude: Center latitude for geographic search.
            longitude: Center longitude for geographic search.
            max_radius_km: Search radius in kilometers.
            limit: Maximum number of results.

        Returns:
            Earthquake catalog results.
        """
        cache_key = f"usgs:eq:{min_magnitude}:{start_time}:{latitude}:{longitude}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        params = {
            "format": "geojson",
            "minmagnitude": min_magnitude,
            "limit": limit,
            "orderby": "time",
        }
        if start_time:
            params["starttime"] = start_time
        if end_time:
            params["endtime"] = end_time
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
            params["maxradiuskm"] = max_radius_km

        result = {
            "source": "usgs_earthquake",
            "api_url": self.EARTHQUAKE_URL,
            "params": params,
            "earthquakes": [],
            "note": "Live data requires network access. Parameters prepared for USGS FDSN API.",
        }

        self._cache.put(cache_key, result)
        return result

    def get_earthquake(self, event_id: str) -> Dict[str, Any]:
        """Fetch details for a specific earthquake event.

        Args:
            event_id: USGS event ID.

        Returns:
            Earthquake event details.
        """
        cache_key = f"usgs:event:{event_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = {
            "source": "usgs_earthquake",
            "event_id": event_id,
            "url": f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}",
            "api_url": f"{self.EARTHQUAKE_URL}?eventid={event_id}&format=geojson",
            "note": "Event details require live API access.",
        }

        self._cache.put(cache_key, result)
        return result

    def available_datasets(self) -> List[Dict[str, str]]:
        """List available USGS data categories."""
        return [
            {"id": "earthquake", "name": "Earthquake Catalog", "api": self.EARTHQUAKE_URL},
            {"id": "water", "name": "Water Services", "api": self.WATER_URL},
            {"id": "geology", "name": "National Geologic Map Database", "api": "https://ngmdb.usgs.gov"},
            {"id": "landsat", "name": "Landsat Satellite Imagery", "api": "https://earthexplorer.usgs.gov"},
            {"id": "elevation", "name": "National Elevation Dataset", "api": "https://apps.nationalmap.gov"},
        ]
