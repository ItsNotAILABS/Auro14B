"""Caching layer for downloaded datasets."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheConfig:
    """Configuration for the data cache.

    Attributes:
        cache_dir: Directory for cached files.
        max_age_seconds: Maximum age before cache entry expires.
        max_size_mb: Maximum cache size in megabytes.
    """

    cache_dir: str = ".mesie_cache"
    max_age_seconds: float = 86400.0  # 24 hours
    max_size_mb: float = 500.0


class DataCache:
    """Simple file-based cache for downloaded datasets.

    Caches API responses and downloaded files with TTL-based expiration.

    Args:
        config: Cache configuration.
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self._config = config or CacheConfig()
        self._cache_dir = Path(self._config.cache_dir)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def _key_hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached entry if it exists and is not expired."""
        entry = self._metadata.get(key)
        if entry is None:
            return None
        if time.time() - entry.get("timestamp", 0) > self._config.max_age_seconds:
            self.invalidate(key)
            return None
        return entry.get("data")

    def put(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in cache."""
        self._metadata[key] = {
            "data": data,
            "timestamp": time.time(),
            "key_hash": self._key_hash(key),
        }

    def invalidate(self, key: str) -> None:
        """Remove a cache entry."""
        self._metadata.pop(key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._metadata.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._metadata)

    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "entries": self.size,
            "cache_dir": str(self._cache_dir),
            "max_age_seconds": self._config.max_age_seconds,
        }
