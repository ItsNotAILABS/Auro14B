"""Spatial lexical memory and executive composition for HIM/AURO.

This module gives small and undertrained model lanes a deterministic language
substrate: every observed word is normalized, decomposed into letters and
n-grams, placed in a stable spatial coordinate system, connected to neighboring
concepts, and made available to retrieval, creative recombination, direct-answer
composition, repetition control, and evidence-aware revision.

It does not pretend to replace learned language modeling. It is an auditable
algorithmic support layer that improves grounding and composition while larger
checkpoints mature.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def words(text: str) -> list[str]:
    return [m.group(0).casefold() for m in _WORD.finditer(normalize_text(text))]


def stable_vector(token: str, dimensions: int = 24) -> tuple[float, ...]:
    """Map spelling, letters, and n-grams into a deterministic unit vector."""
    token = normalize_text(token).casefold()
    features: list[str] = [f"w:{token}"]
    features.extend(f"c:{i}:{c}" for i, c in enumerate(token))
    padded = f"^{token}$"
    for n in (2, 3, 4):
        features.extend(f"g{n}:{padded[i:i+n]}" for i in range(max(0, len(padded) - n + 1)))
    vec = [0.0] * dimensions
    for feature in features:
        raw = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        for i, byte in enumerate(raw):
            vec[i % dimensions] += (byte / 127.5) - 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return tuple(x / norm for x in vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class Lexeme:
    token: str
    vector: tuple[float, ...]
    count: int = 0
    documents: set[str] = field(default_factory=set)
    neighbors: Counter[str] = field(default_factory=Counter)
    forms: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "count": self.count,
            "documents": sorted(self.documents),
            "neighbors": dict(self.neighbors.most_common(20)),
            "forms": dict(self.forms.most_common(10)),
            "vector": list(self.vector),
        }


class SpatialLexicon:
    """A persistent-in-process word/letter library with spatial retrieval."""

    def __init__(self, dimensions: int = 24, window: int = 4) -> None:
        self.dimensions = dimensions
        self.window = window
        self.lexemes: dict[str, Lexeme] = {}
        self.documents: dict[str, str] = {}
        self.document_terms: dict[str, Counter[str]] = {}

    def ingest(self, text: str, *, source: str = "memory") -> dict[str, Any]:
        clean = normalize_text(text)
        source_id = hashlib.sha256(f"{source}\0{clean}".encode()).hexdigest()[:20]
        tokens = words(clean)
        self.documents[source_id] = clean
        counts = Counter(tokens)
        self.document_terms[source_id] = counts
        for index, token in enumerate(tokens):
            lexeme = self.lexemes.setdefault(token, Lexeme(token, stable_vector(token, self.dimensions)))
            lexeme.count += 1
            lexeme.documents.add(source_id)
            lexeme.forms[token] += 1
            left = max(0, index - self.window)
            right = min(len(tokens), index + self.window + 1)
            for neighbor in tokens[left:index] + tokens[index + 1:right]:
                if neighbor != token:
                    lexeme.neighbors[neighbor] += 1
        return {
            "source_id": source_id,
            "tokens": len(tokens),
            "unique_tokens": len(counts),
            "lexicon_size": len(self.lexemes),
        }

    def retrieve(self, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        query_terms = words(query)
        if not query_terms:
            return []
        q_vectors = [stable_vector(term, self.dimensions) for term in query_terms]
        qv = tuple(sum(v[i] for v in q_vectors) / len(q_vectors) for i in range(self.dimensions))
        qset = set(query_terms)
        scored: list[tuple[float, str]] = []
        for doc_id, counts in self.document_terms.items():
            overlap = sum(counts[t] for t in qset)
            semantic = 0.0
            total = sum(counts.values()) or 1
            for term, count in counts.most_common(64):
                semantic += max(0.0, cosine(qv, self.lexemes[term].vector)) * (count / total)
            score = overlap * 2.5 + semantic
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(reverse=True)
        return [
            {
                "source_id": doc_id,
                "score": round(score, 6),
                "text": self.documents[doc_id],
                "terms": dict(self.document_terms[doc_id].most_common(12)),
            }
            for score, doc_id in scored[:top_k]
        ]

    def associations(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        query_terms = words(query)
        if not query_terms:
            return []
        qv = stable_vector(" ".join(query_terms), self.dimensions)
        direct = Counter()
        for term in query_terms:
            if term in self.lexemes:
                direct.update(self.lexemes[term].neighbors)
        candidates = set(direct)
        candidates.update(self.lexemes)
        ranked = []
        for token in candidates:
            lex = self.lexemes[token]
            score = cosine(qv, lex.vector) + math.log1p(direct[token]) * 0.35 + math.log1p(lex.count) * 0.05
            if token not in query_terms:
                ranked.append((score, token, direct[token]))
        ranked.sort(reverse=True)
        return [
            {"token": token, "score": round(score, 6), "cooccurrence": cooccurrence}
            for score, token, cooccurrence in ranked[:top_k]
        ]

    def manifest(self) -> dict[str, Any]:
        material = {token: lexeme.to_dict() for token, lexeme in sorted(self.lexemes.items())}
        digest = hashlib.sha256(
            repr([(token, item["count"], item["documents"]) for token, item in material.items()]).encode()
        ).hexdigest()
        return {
            "schema": "auro.spatial-lexicon.v1",
            "dimensions": self.dimensions,
            "window": self.window,
            "documents": len(self.documents),
            "lexemes": len(self.lexemes),
            "sha256": digest,
        }


class ExecutiveComposer:
    """Compose direct, evidence-aware answers from candidate artifacts."""

    UNCERTAINTY = ("unknown", "cannot verify", "not inspected", "unconfirmed", "requires evidence")

    def __init__(self, lexicon: SpatialLexicon | None = None) -> None:
        self.lexicon = lexicon or SpatialLexicon()

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", normalize_text(text)) if p.strip()]

    @staticmethod
    def _signature(text: str) -> set[str]:
        return set(words(text))

    def deduplicate(self, text: str, threshold: float = 0.82) -> str:
        kept: list[str] = []
        signatures: list[set[str]] = []
        for paragraph in self._paragraphs(text):
            sig = self._signature(paragraph)
            duplicate = False
            for previous in signatures:
                union = len(sig | previous) or 1
                if len(sig & previous) / union >= threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(paragraph)
                signatures.append(sig)
        return "\n\n".join(kept)

    def score(self, prompt: str, answer: str) -> dict[str, float]:
        p = set(words(prompt))
        a = words(answer)
        aset = set(a)
        relevance = len(p & aset) / max(1, len(p))
        sentences = [s for s in _SENTENCE.split(answer) if s.strip()]
        opening = set(words(sentences[0] if sentences else answer[:240]))
        directness = len(p & opening) / max(1, min(len(p), 12))
        repetition = 0.0
        if a:
            counts = Counter(a)
            repetition = sum(max(0, count - 3) for count in counts.values()) / len(a)
        uncertainty = float(any(marker in answer.casefold() for marker in self.UNCERTAINTY))
        return {
            "relevance": round(relevance, 4),
            "directness": round(directness, 4),
            "repetition": round(repetition, 4),
            "uncertainty_marked": uncertainty,
        }

    def compose(
        self,
        prompt: str,
        candidates: Iterable[str],
        *,
        evidence: Iterable[str] = (),
        require_uncertainty: bool = False,
    ) -> dict[str, Any]:
        candidates = [normalize_text(c) for c in candidates if normalize_text(c)]
        evidence = [normalize_text(e) for e in evidence if normalize_text(e)]
        for i, item in enumerate((*candidates, *evidence)):
            self.lexicon.ingest(item, source=f"compose:{i}")
        ranked = []
        for candidate in candidates:
            metrics = self.score(prompt, candidate)
            aggregate = metrics["relevance"] * 0.5 + metrics["directness"] * 0.35 - metrics["repetition"] * 0.4
            if require_uncertainty:
                aggregate += metrics["uncertainty_marked"] * 0.25
            ranked.append((aggregate, candidate, metrics))
        ranked.sort(key=lambda row: row[0], reverse=True)
        selected = ranked[0][1] if ranked else "I do not yet have enough grounded material to answer this directly."
        selected = self.deduplicate(selected)
        if require_uncertainty and not any(marker in selected.casefold() for marker in self.UNCERTAINTY):
            selected = "I cannot verify this from the evidence currently available.\n\n" + selected
        associations = self.lexicon.associations(prompt, top_k=12)
        return {
            "text": selected,
            "score": self.score(prompt, selected),
            "candidate_count": len(candidates),
            "evidence_count": len(evidence),
            "associations": associations,
            "lexicon": self.lexicon.manifest(),
        }

    def creative_scaffold(self, prompt: str, *, branches: int = 6) -> dict[str, Any]:
        associations = self.lexicon.associations(prompt, top_k=max(branches * 2, 12))
        terms = [row["token"] for row in associations]
        lenses = ("mechanism", "counterfactual", "analogy", "constraint", "failure", "synthesis")
        seeds = []
        for index in range(branches):
            related = terms[index:index + 3]
            seeds.append({
                "lens": lenses[index % len(lenses)],
                "related_terms": related,
                "instruction": f"Explore {prompt!r} through {lenses[index % len(lenses)]} using {', '.join(related) or 'first principles'}.",
            })
        return {"prompt": prompt, "branches": seeds, "lexicon": self.lexicon.manifest()}
