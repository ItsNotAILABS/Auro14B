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
    admission_rejections: int = 0


class ExpertPager:
    """Hot set over an ``ExpertColdStore`` with a byte budget.

    Two eviction policies:

    * ``lru`` -- textbook default. Measured finding (2026-09-10,
      ``test_lru_thrashes_on_cyclic_scan``): pathologically misses on cyclic
      expert-firing patterns longer than the hot set.
    * ``tinylfu`` -- TinyLFU-style admission filter over a frequency sketch.
      On a demand miss that would require an eviction, the newcomer is only
      admitted if it has fired more often than the LRU victim; otherwise its
      weights are loaded transiently for the forward pass but not kept hot.
      This pins genuinely hot experts and beats LRU on skewed patterns
      (``test_tinylfu_pins_hot_expert``).

    Prefetch never evicts under either policy: it only spends headroom.
    """

    # How many demand accesses between frequency-sketch aging halves. Aging
    # keeps the sketch adaptive: yesterday's hot expert does not pin forever.
    SKETCH_RESET_INTERVAL = 1024

    def __init__(self, store: ExpertColdStore, budget_bytes: int, policy: str = "lru"):
        if budget_bytes <= 0:
            raise ValueError("budget_bytes must be positive")
        if policy not in ("lru", "tinylfu"):
            raise ValueError(f"unknown eviction policy: {policy}")
        self.store = store
        self.budget_bytes = int(budget_bytes)
        self.policy = policy
        self._hot: Dict[Key, _HotEntry] = {}
        self._freq: Dict[Key, int] = {}
        self._freq_ops = 0
        self._tick = 0
        self.stats = PagerStats()

    def _touch(self, key: Key) -> None:
        self._tick += 1
        self._hot[key].last_used = self._tick

    def _record_use(self, key: Key) -> None:
        """Count a demand access in the frequency sketch, aging periodically."""
        self._freq[key] = self._freq.get(key, 0) + 1
        self._freq_ops += 1
        if self._freq_ops >= self.SKETCH_RESET_INTERVAL:
            self._freq = {k: v // 2 for k, v in self._freq.items() if v // 2}
            self._freq_ops = 0

    def _admit(self, key: Key, need: int) -> bool:
        """TinyLFU admission: admit if it fits, else only if hotter than the victim."""
        if self.stats.hot_bytes + need <= self.budget_bytes:
            return True
        if self.policy != "tinylfu":
            return True
        victim = min(self._hot.items(), key=lambda kv: kv[1].last_used)[0]
        return self._freq.get(key, 0) > self._freq.get(victim, 0)

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

    def _load_cold(self, layer: int, expert: int, prefetched: bool) -> _HotEntry | Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        weights = self.store.load(layer, expert, mmap=True)
        stall_ms = (time.perf_counter() - t0) * 1000.0
        nbytes = self.store.expert_nbytes(layer, expert)
        key = (layer, expert)
        if not prefetched and not self._admit(key, nbytes):
            # Admission rejected: weights are still needed for this forward
            # pass, so serve them transiently without caching. Bytes moved and
            # stall are counted honestly; the hot set stays clean.
            self.stats.misses += 1
            self.stats.bytes_moved += nbytes
            self.stats.demand_stall_ms += stall_ms
            self.stats.admission_rejections += 1
            return weights
        self._evict_until_fits(nbytes)
        if self.stats.hot_bytes + nbytes > self.budget_bytes:
            self.stats.over_budget_loads += 1
        entry = _HotEntry(weights=weights, nbytes=nbytes, prefetched=prefetched)
        self._hot[key] = entry
        self._touch(key)
        self.stats.hot_bytes += nbytes
        self.stats.bytes_moved += nbytes
        self.stats.peak_hot_bytes = max(self.stats.peak_hot_bytes, self.stats.hot_bytes)
        if not prefetched:
            self.stats.misses += 1
            self.stats.demand_stall_ms += stall_ms
        else:
            self.stats.prefetch_issued += 1
        return entry

    def acquire(self, layer: int, expert: int) -> Dict[str, np.ndarray]:
        """Get an expert's weights, paging it in on miss. Returns the arrays."""
        key = (layer, expert)
        self._record_use(key)
        entry = self._hot.get(key)
        if entry is not None:
            self.stats.hits += 1
            if entry.prefetched:
                self.stats.prefetch_hits += 1
                entry.prefetched = False
            self._touch(key)
            return entry.weights
        loaded = self._load_cold(layer, expert, prefetched=False)
        if isinstance(loaded, _HotEntry):
            return loaded.weights
        return loaded

    def prefetch(self, layer: int, expert: int) -> bool:
        """Warm an expert ahead of demand. Returns True if it caused a load.

        Prefetch never evicts: it only spends headroom, so a wrong prediction
        wastes bytes but cannot thrash the hot set.
        """
        if (layer, expert) in self._hot:
            return False
        need = self.store.expert_nbytes(layer, expert)
        if self.stats.hot_bytes + need > self.budget_bytes:
            return False
        loaded = self._load_cold(layer, expert, prefetched=True)
        assert isinstance(loaded, _HotEntry)
        return True

    def snapshot(self) -> Dict[str, object]:
        s = self.stats
        total_demand = s.hits + s.misses
        return {
            "budget_bytes": self.budget_bytes,
            "eviction_policy": self.policy,
            "hits": s.hits,
            "misses": s.misses,
            "hit_rate": (s.hits / total_demand) if total_demand else 0.0,
            "evictions": s.evictions,
            "admission_rejections": s.admission_rejections,
            "bytes_moved": s.bytes_moved,
            "prefetch_issued": s.prefetch_issued,
            "prefetch_hits": s.prefetch_hits,
            "prefetch_hit_rate": (s.prefetch_hits / s.prefetch_issued) if s.prefetch_issued else 0.0,
            "demand_stall_ms": round(s.demand_stall_ms, 3),
            "hot_bytes": s.hot_bytes,
            "peak_hot_bytes": s.peak_hot_bytes,
            "over_budget_loads": s.over_budget_loads,
        }
