"""Chemistry Lab — molecular fingerprinting, reaction networks, and compound analysis."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult


class ChemistryLab(BaseLab):
    """Lab for computational chemistry and molecular analysis.

    Capabilities include molecular fingerprinting, similarity search,
    property prediction, and reaction network analysis.
    """

    def _default_config(self) -> LabConfig:
        return LabConfig(
            name="Chemistry Lab",
            domain="chemistry",
            capabilities=[
                "molecular_fingerprint",
                "similarity_search",
                "property_predict",
                "reaction_network",
                "compound_lookup",
                "formula_parse",
            ],
        )

    def run(self, operation: str, **kwargs: Any) -> LabResult:
        start = time.time()
        try:
            if operation == "molecular_fingerprint":
                data = self._molecular_fingerprint(**kwargs)
            elif operation == "similarity_search":
                data = self._similarity_search(**kwargs)
            elif operation == "property_predict":
                data = self._property_predict(**kwargs)
            elif operation == "reaction_network":
                data = self._reaction_network(**kwargs)
            elif operation == "compound_lookup":
                data = self._compound_lookup(**kwargs)
            elif operation == "formula_parse":
                data = self._formula_parse(**kwargs)
            else:
                return LabResult(
                    lab=self.name, operation=operation,
                    status="error", error=f"Unknown operation: {operation}",
                )
            return LabResult(
                lab=self.name, operation=operation, data=data,
                duration_seconds=time.time() - start,
            )
        except Exception as exc:
            return LabResult(
                lab=self.name, operation=operation,
                status="error", error=str(exc),
                duration_seconds=time.time() - start,
            )

    def _molecular_fingerprint(
        self, smiles: str = "", bits: int = 1024, **kw: Any
    ) -> Dict[str, Any]:
        """Generate a binary fingerprint vector from SMILES notation."""
        if not smiles:
            return {"error": "SMILES string required"}
        # Deterministic hash-based fingerprint (no RDKit dependency)
        rng = np.random.default_rng(
            int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
        )
        fp = rng.integers(0, 2, size=bits).tolist()
        return {
            "smiles": smiles,
            "bits": bits,
            "fingerprint_ones": sum(fp),
            "density": sum(fp) / bits,
        }

    def _similarity_search(
        self, query_smiles: str = "", database: Optional[List[str]] = None, top_k: int = 5, **kw: Any
    ) -> Dict[str, Any]:
        """Tanimoto similarity search against a compound database."""
        if not query_smiles:
            return {"error": "query_smiles required"}
        db = database or ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)O", "C(=O)O"]
        # Compute fingerprints and Tanimoto coefficients
        query_fp = self._hash_fingerprint(query_smiles, 256)
        results = []
        for smi in db:
            candidate_fp = self._hash_fingerprint(smi, 256)
            tanimoto = self._tanimoto(query_fp, candidate_fp)
            results.append({"smiles": smi, "tanimoto": round(tanimoto, 4)})
        results.sort(key=lambda x: x["tanimoto"], reverse=True)
        return {"query": query_smiles, "results": results[:top_k]}

    def _property_predict(
        self, smiles: str = "", properties: Optional[List[str]] = None, **kw: Any
    ) -> Dict[str, Any]:
        """Predict molecular properties from structure (heuristic model)."""
        if not smiles:
            return {"error": "SMILES string required"}
        props = properties or ["molecular_weight", "logP", "num_atoms"]
        # Simple heuristic estimates
        n_heavy = len([c for c in smiles if c.isalpha() and c.isupper()])
        predictions = {}
        if "molecular_weight" in props:
            predictions["molecular_weight"] = round(n_heavy * 14.0 + 2.0, 1)
        if "logP" in props:
            predictions["logP"] = round((n_heavy - 2) * 0.5, 2)
        if "num_atoms" in props:
            predictions["num_atoms"] = n_heavy
        return {"smiles": smiles, "predictions": predictions}

    def _reaction_network(
        self, reactants: Optional[List[str]] = None, **kw: Any
    ) -> Dict[str, Any]:
        """Construct a simple reaction network from reactants."""
        reactants = reactants or []
        nodes = [{"id": i, "smiles": r} for i, r in enumerate(reactants)]
        # Simple pairwise edges
        edges = [
            {"from": i, "to": j, "type": "potential_reaction"}
            for i in range(len(reactants))
            for j in range(i + 1, len(reactants))
        ]
        return {"nodes": nodes, "edges": edges, "n_reactions": len(edges)}

    def _compound_lookup(self, name: str = "", **kw: Any) -> Dict[str, Any]:
        """Look up compound info by name (built-in mini-database)."""
        db = {
            "water": {"formula": "H2O", "mw": 18.015, "smiles": "O"},
            "ethanol": {"formula": "C2H5OH", "mw": 46.069, "smiles": "CCO"},
            "acetic acid": {"formula": "CH3COOH", "mw": 60.052, "smiles": "CC(=O)O"},
            "benzene": {"formula": "C6H6", "mw": 78.114, "smiles": "c1ccccc1"},
            "glucose": {"formula": "C6H12O6", "mw": 180.156, "smiles": "OC[C@H]1OC(O)C(O)C(O)C1O"},
        }
        entry = db.get(name.lower())
        if entry:
            return {"name": name, **entry}
        return {"error": f"Compound '{name}' not in local database"}

    def _formula_parse(self, formula: str = "", **kw: Any) -> Dict[str, Any]:
        """Parse a chemical formula into element counts."""
        if not formula:
            return {"error": "Formula string required"}
        import re
        elements: Dict[str, int] = {}
        for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
            elem, count = match.groups()
            if elem:
                elements[elem] = elements.get(elem, 0) + int(count or 1)
        total_atoms = sum(elements.values())
        return {"formula": formula, "elements": elements, "total_atoms": total_atoms}

    # --- Helpers ---

    def _hash_fingerprint(self, smiles: str, bits: int) -> np.ndarray:
        rng = np.random.default_rng(
            int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
        )
        return rng.integers(0, 2, size=bits)

    def _tanimoto(self, a: np.ndarray, b: np.ndarray) -> float:
        intersection = float(np.sum(a & b))
        union = float(np.sum(a | b))
        return intersection / union if union > 0 else 0.0
