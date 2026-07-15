"""Bio Lab — biosignal processing, sequence analysis, and genomics utilities."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from mesie.labs.base_lab import BaseLab, LabConfig, LabResult


# Standard genetic code
CODON_TABLE: Dict[str, str] = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


class BioLab(BaseLab):
    """Lab for bioinformatics, biosignal analysis, and genomics.

    Capabilities include DNA/RNA sequence analysis, protein translation,
    sequence alignment scoring, biosignal filtering, and GC content analysis.
    """

    def _default_config(self) -> LabConfig:
        return LabConfig(
            name="Bio Lab",
            domain="bio",
            capabilities=[
                "gc_content",
                "translate",
                "complement",
                "sequence_stats",
                "alignment_score",
                "biosignal_filter",
                "kmer_frequency",
            ],
        )

    def run(self, operation: str, **kwargs: Any) -> LabResult:
        start = time.time()
        try:
            if operation == "gc_content":
                data = self._gc_content(**kwargs)
            elif operation == "translate":
                data = self._translate(**kwargs)
            elif operation == "complement":
                data = self._complement(**kwargs)
            elif operation == "sequence_stats":
                data = self._sequence_stats(**kwargs)
            elif operation == "alignment_score":
                data = self._alignment_score(**kwargs)
            elif operation == "biosignal_filter":
                data = self._biosignal_filter(**kwargs)
            elif operation == "kmer_frequency":
                data = self._kmer_frequency(**kwargs)
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

    def _gc_content(self, sequence: str = "", **kw: Any) -> Dict[str, Any]:
        """Compute GC content of a DNA/RNA sequence."""
        if not sequence:
            return {"error": "Sequence required"}
        seq = sequence.upper()
        gc = sum(1 for b in seq if b in "GC")
        total = len(seq)
        return {
            "sequence_length": total,
            "gc_count": gc,
            "gc_content": round(gc / total, 4) if total > 0 else 0.0,
        }

    def _translate(self, rna_sequence: str = "", **kw: Any) -> Dict[str, Any]:
        """Translate RNA sequence to amino acid sequence."""
        if not rna_sequence:
            return {"error": "RNA sequence required"}
        seq = rna_sequence.upper().replace("T", "U")
        protein = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            aa = CODON_TABLE.get(codon, "?")
            if aa == "*":
                break
            protein.append(aa)
        return {
            "rna_length": len(seq),
            "protein": "".join(protein),
            "protein_length": len(protein),
        }

    def _complement(self, sequence: str = "", reverse: bool = True, **kw: Any) -> Dict[str, Any]:
        """Get the (reverse) complement of a DNA sequence."""
        if not sequence:
            return {"error": "Sequence required"}
        comp_map = str.maketrans("ATCGatcg", "TAGCtagc")
        comp = sequence.translate(comp_map)
        if reverse:
            comp = comp[::-1]
        return {
            "original": sequence,
            "complement": comp,
            "reverse": reverse,
        }

    def _sequence_stats(self, sequence: str = "", **kw: Any) -> Dict[str, Any]:
        """Basic statistics for a nucleotide sequence."""
        if not sequence:
            return {"error": "Sequence required"}
        seq = sequence.upper()
        counts = {base: seq.count(base) for base in "ATCGU"}
        counts = {k: v for k, v in counts.items() if v > 0}
        return {
            "length": len(seq),
            "base_counts": counts,
            "gc_content": round(
                (counts.get("G", 0) + counts.get("C", 0)) / len(seq), 4
            )
            if len(seq) > 0
            else 0.0,
        }

    def _alignment_score(
        self, seq_a: str = "", seq_b: str = "", match: int = 1, mismatch: int = -1, gap: int = -2, **kw: Any
    ) -> Dict[str, Any]:
        """Simple global alignment score (Needleman-Wunsch style scoring)."""
        if not seq_a or not seq_b:
            return {"error": "Both seq_a and seq_b required"}
        # Simplified scoring without full DP (just positional comparison)
        min_len = min(len(seq_a), len(seq_b))
        score = 0
        matches = 0
        for i in range(min_len):
            if seq_a[i].upper() == seq_b[i].upper():
                score += match
                matches += 1
            else:
                score += mismatch
        # Gap penalty for length difference
        score += abs(len(seq_a) - len(seq_b)) * gap
        identity = matches / min_len if min_len > 0 else 0.0
        return {
            "score": score,
            "identity": round(identity, 4),
            "matches": matches,
            "length_a": len(seq_a),
            "length_b": len(seq_b),
        }

    def _biosignal_filter(
        self,
        signal: Optional[List[float]] = None,
        filter_type: str = "lowpass",
        cutoff_hz: float = 40.0,
        sample_rate: float = 250.0,
        **kw: Any,
    ) -> Dict[str, Any]:
        """Apply a simple moving-average filter to a biosignal."""
        if signal is None or len(signal) == 0:
            return {"error": "Signal array required"}
        arr = np.array(signal, dtype=float)
        # Window size based on cutoff
        window = max(1, int(sample_rate / cutoff_hz))
        if filter_type == "lowpass":
            kernel = np.ones(window) / window
            filtered = np.convolve(arr, kernel, mode="same")
        elif filter_type == "highpass":
            kernel = np.ones(window) / window
            low = np.convolve(arr, kernel, mode="same")
            filtered = arr - low
        else:
            filtered = arr
        return {
            "filter_type": filter_type,
            "cutoff_hz": cutoff_hz,
            "sample_rate": sample_rate,
            "input_length": len(signal),
            "output_mean": float(np.mean(filtered)),
            "output_std": float(np.std(filtered)),
        }

    def _kmer_frequency(self, sequence: str = "", k: int = 3, **kw: Any) -> Dict[str, Any]:
        """Compute k-mer frequencies for a sequence."""
        if not sequence:
            return {"error": "Sequence required"}
        seq = sequence.upper()
        kmers: Dict[str, int] = {}
        for i in range(len(seq) - k + 1):
            kmer = seq[i : i + k]
            kmers[kmer] = kmers.get(kmer, 0) + 1
        total = sum(kmers.values())
        top_5 = sorted(kmers.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "k": k,
            "total_kmers": total,
            "unique_kmers": len(kmers),
            "top_5": [{"kmer": km, "count": c} for km, c in top_5],
        }
