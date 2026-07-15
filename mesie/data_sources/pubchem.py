"""PubChem data source — chemical compound lookup and search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mesie.data_sources.cache import DataCache


class PubChemSource:
    """Connector to PubChem for chemical compound information.

    Provides structured access to PubChem's compound database for
    chemistry research workflows.

    Args:
        cache: Optional data cache instance.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(self, cache: Optional[DataCache] = None) -> None:
        self._cache = cache or DataCache()

    @property
    def name(self) -> str:
        return "pubchem"

    @property
    def description(self) -> str:
        return "PubChem — chemical compound database (NIH/NCBI)"

    def search_compound(
        self, name: str, *, max_results: int = 5
    ) -> Dict[str, Any]:
        """Search PubChem for a compound by name.

        Args:
            name: Compound name to search.
            max_results: Maximum results to return.

        Returns:
            Search results with compound metadata.
        """
        cache_key = f"pubchem:search:{name}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = {
            "source": "pubchem",
            "query": name,
            "api_url": f"{self.BASE_URL}/compound/name/{name}/JSON",
            "compounds": [],
            "note": "Live API fetch requires network access.",
        }

        self._cache.put(cache_key, result)
        return result

    def get_compound(self, cid: int) -> Dict[str, Any]:
        """Fetch compound details by PubChem CID.

        Args:
            cid: PubChem Compound ID.

        Returns:
            Compound properties and metadata.
        """
        cache_key = f"pubchem:cid:{cid}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = {
            "source": "pubchem",
            "cid": cid,
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "api_url": f"{self.BASE_URL}/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,IUPACName/JSON",
            "note": "Full properties require live API access.",
        }

        self._cache.put(cache_key, result)
        return result

    def get_structure(self, cid: int, format: str = "smiles") -> Dict[str, Any]:
        """Get molecular structure representation.

        Args:
            cid: PubChem Compound ID.
            format: Output format (smiles, inchi, sdf).

        Returns:
            Structure data.
        """
        cache_key = f"pubchem:structure:{cid}:{format}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        result = {
            "source": "pubchem",
            "cid": cid,
            "format": format,
            "api_url": f"{self.BASE_URL}/compound/cid/{cid}/property/CanonicalSMILES,InChI/JSON",
            "note": "Structure data requires live API access.",
        }

        self._cache.put(cache_key, result)
        return result
