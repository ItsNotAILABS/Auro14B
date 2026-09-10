"""Demand-paged expert hot set with a byte budget and prefetch.

``ExpertPager`` is the "weights stay stale, load only when needed" mechanism:
the router fires experts, the pager makes them hot. Hits, misses, evictions,
bytes moved and stall milliseconds are all counted so the run's memory story
is measured, not asserted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .cold_store import ExpertColdStore

Key = Tuple[int, int]


@dataclass
class _HotEntry:
    weights: Dict[str, np.ndarray]
    nbytes: int
    last_used: int = 0
    prefetched: bool = False


@dataclass
class PagerStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_moved: int = 0
    prefetch_issued: int = 0
    prefetch_hits: int = 0
    demand_stall_ms: float = 0.0
    hot_bytes: int = 0
    peak_hot_bytes: int = 0
    over_budget_loads: int = 0


class ExpertPager:
    """LRU hot set over an ``ExpertColdStore`` with a byte budget.

    Measured finding (2026-09-10, ``test_lru_thrashes_on_cyclic_scan``): LRU
    pathologically misses on cyclic expert-firing patterns longer than the
    hot set. That is an honest measurement, not a bug -- and it is why the
    eviction policy is the next thing to beat, e.g. with a TinyLFU-style
    frequency-sketch admission filter. The bench exists to measure that race.
    """

    def __init__(self, store: ExpertColdStore, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        self.store = store
        self.budget_bytes = int(budget_bytes)
        self._hot: Dict[Key, _HotEntry] = {}
        self._tick = 0
        self.stats = PagerStats()

    def _touch(self, key: Key) -> None:
        self._tick += 1
        self._hot[key].last_used = self._tick

    def _evict_until_fits(self, need: int) -> None:
        while self._hot and self.stats.hot_bytes + need > self.budget_bytes:
            victim = min(self._hot.items(), key=lambda kv: kv[1].last_used)[0]
            entry = self._hot.pop(victim)
            # release mmap handles so the OS can reclaim pages
            for arr in entry.weights.values():
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()
            self.stats.hot_bytes -= entry.nbytes
            self.stats.evictions += 1

    def _load_cold(self, layer: int, expert: int, prefetched: bool) -> _HotEntry:
        t0 = time.perf_counter()
        weights = self.store.load(layer, expert, mmap=True)
        stall_ms = (time.perf_counter() - t0) * 1000.0
        nbytes = self.store.expert_nbytes(layer, expert)
        self._evict_until_fits(nbytes)
        if self.stats.hot_bytes + nbytes > self.budget_bytes:
            self.stats.over_budget_loads += 1
        entry = _HotEntry(weights=weights, nbytes=nbytes, prefetched=prefetched)
        self._hot[(layer, expert)] = entry
        self._touch((layer, expert))
        self.stats.hot_bytes += nbytes
        self.stats.bytes_moved += nbytes
        self.stats.peak_hot_bytes = max(self.stats.peak_hot_bytes, self.stats.hot_bytes)
        if not prefetched:
            self.stats.misses += 1
            self.stats.demand_stall_ms += stall_ms
        return entry

    def acquire(self, layer: int, expert: int) -> Dict[str, np.ndarray]:
        """Get an expert's weights, paging it in on miss. Returns the arrays."""
        key = (layer, expert)
        entry = self._hot.get(key)
        if entry is not None:
            self.stats.hits += 1
            if entry.prefetched:
                self.stats.prefetch_hits += 1
                entry.prefetched = False
            self._touch(key)
            return entry.weights
        return self._load_cold(layer, expert, prefetched=False).weights

    def prefetch(self, layer: int, expert: int) -> bool:
        """Warm an expert ahead of demand. Returns True if it caused a load."""
        if (layer, expert) in self._hot:
            return False
        self._load_cold(layer, expert, prefetched=True)
        self.stats.prefetch_issued += 1
        return True

    def snapshot(self) -> Dict[str, object]:
        s = self.stats
        total_demand = s.hits + s.misses
        return {
            "budget_bytes": self.budget_bytes,
            "eviction_policy": "lru",
            "hits": s.hits,
            "misses": s.misses,
            "hit_rate": (s.hits / total_demand) if total_demand else 0.0,
            "evictions": s.evictions,
            "bytes_moved": s.bytes_moved,
            "prefetch_issued": s.prefetch_issued,
            "prefetch_hits": s.prefetch_hits,
            "prefetch_hit_rate": (s.prefetch_hits / s.prefetch_issued) if s.prefetch_issued else 0.0,
            "demand_stall_ms": round(s.demand_stall_ms, 3),
            "hot_bytes": s.hot_bytes,
            "peak_hot_bytes": s.peak_hot_bytes,
            "over_budget_loads": s.over_budget_loads,
        }
