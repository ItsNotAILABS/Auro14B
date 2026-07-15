"""ArXiv data source — search and fetch scientific papers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.data_sources.cache import DataCache


class ArxivSource:
    """Connector to ArXiv for searching and retrieving paper metadata.

    Provides structured access to ArXiv's paper database for literature
    reviews and citation management.

    Args:
        cache: Optional data cache instance.
        max_results: Default maximum results per query.
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self, cache: Optional[DataCache] = None, max_results: int = 10
    ) -> None:
        self._cache = cache or DataCache()
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def description(self) -> str:
        return "ArXiv preprint server — physics, math, CS, biology papers"

    def search(
        self,
        query: str,
        *,
        max_results: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search ArXiv for papers matching a query.

        Args:
            query: Search query string.
            max_results: Override default max results.
            category: Optional ArXiv category filter (e.g., 'physics.optics').

        Returns:
            Dict with search results metadata.
        """
        cache_key = f"arxiv:search:{query}:{category}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Construct query parameters (actual HTTP fetch would go here)
        search_params = {
            "search_query": f"all:{query}" + (f"+AND+cat:{category}" if category else ""),
            "max_results": max_results or self._max_results,
        }

        # Simulated response structure (real impl would use urllib/requests)
        result = {
            "source": "arxiv",
            "query": query,
            "category": category,
            "params": search_params,
            "total_results": 0,
            "papers": [],
            "api_url": f"{self.BASE_URL}?search_query={search_params['search_query']}",
            "note": "Live API fetch requires network access. Use mesie hub with network enabled.",
        }

        self._cache.put(cache_key, result)
        return result

    def fetch_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch metadata for a specific ArXiv paper.

        Args:
            arxiv_id: ArXiv paper ID (e.g., '2301.12345').

        Returns:
            Paper metadata dictionary.
        """
        cache_key = f"arxiv:paper:{arxiv_id}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "note": "Full metadata requires live API access.",
        }

        self._cache.put(cache_key, result)
        return result

    def categories(self) -> List[str]:
        """List common ArXiv categories."""
        return [
            "physics.optics",
            "physics.atom-ph",
            "cond-mat.mtrl-sci",
            "cs.AI",
            "cs.LG",
            "q-bio",
            "math.NA",
            "astro-ph",
            "eess.SP",
        ]
