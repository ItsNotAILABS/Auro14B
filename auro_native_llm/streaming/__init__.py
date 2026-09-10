"""Streaming inference prototype: stale-by-default expert paging with receipts.

Pipeline
--------
``ExpertColdStore``  -- expert weights live as per-expert ``.npz`` files on disk
``ExpertPager``      -- LRU hot set with a byte budget + prefetch; counts every
                        hit, miss, eviction, byte moved and stall millisecond
``StreamingRun``     -- per-token expert traces finalized into a signed receipt

The benchmark in ``bench.py`` exercises real ``MixtureOfExperts`` routing and
real expert forward passes. Weights are synthetic (random init) unless the
caller supplies trained ones, so receipts are *mechanics* benchmarks --
routing/paging/memory behavior -- never model-quality claims. The receipt's
``claim_boundary`` says exactly that.
"""
from .cold_store import ExpertColdStore
from .pager import ExpertPager
from .receipts import StreamingRun, finalize_receipt

__all__ = ["ExpertColdStore", "ExpertPager", "StreamingRun", "finalize_receipt"]
