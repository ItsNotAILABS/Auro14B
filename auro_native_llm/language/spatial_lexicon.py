"""Persistent lexical-spatial language support for HIM/AURO.

The engine preserves normalized text, UTF-8 bytes, characters, words, phrases,
exact source offsets, sentence membership and weighted co-occurrence. It is an
auditable support layer, not a substitute for learned language modelling.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip()


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD.finditer(normalize_text(text))]


def stable_vector(token: str, dimensions: int = 48) -> tuple[float, ...]:
    clean = normalize_text(token).casefold()
    raw = clean.encode("utf-8")
    features = [f"w:{clean}", f"len:{len(clean)}", f"bytes:{len(raw)}"]
    features += [f"b:{i}:{byte}" for i, byte in enumerate(raw)]
    features += [f"c:{i}:{char}" for i, char in enumerate(clean)]
    padded = f"^{clean}$"
    for size in (2, 3, 4, 5):
        features += [f"g{size}:{padded[i:i + size]}" for i in range(max(0, len(padded) - size + 1))]
    vector = [0.0] * dimensions
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=32).digest()
        for index, byte in enumerate(digest):
            vector[index % dimensions] += byte / 127.5 - 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return tuple(value / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Occurrence:
    source_id: str
    token_index: int
    sentence_index: int
    start: int
    end: int
    surface: str


@dataclass
class Lexeme:
    token: str
    vector: tuple[float, ...]
    count: int = 0
    documents: set[str] = field(default_factory=set)
    neighbors: Counter[str] = field(default_factory=Counter)
    forms: Counter[str] = field(default_factory=Counter)
    occurrences: list[Occurrence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "vector": list(self.vector),
            "count": self.count,
            "documents": sorted(self.documents),
            "neighbors": dict(self.neighbors),
            "forms": dict(self.forms),
            "occurrences": [asdict(item) for item in self.occurrences],
        }


class SpatialLexicon:
    schema = "auro.spatial-lexicon.v2"

    def __init__(self, dimensions: int = 48, window: int = 5, phrase_sizes: Sequence[int] = (2, 3, 4, 5)) -> None:
        self.dimensions = dimensions
        self.window = window
        self.phrase_sizes = tuple(phrase_sizes)
        self.lexemes: dict[str, Lexeme] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.document_terms: dict[str, Counter[str]] = {}
        self.character_counts: Counter[str] = Counter()
        self.byte_counts: Counter[int] = Counter()
        self.character_ngrams: Counter[str] = Counter()
        self.phrase_counts: Counter[str] = Counter()
        self.total_tokens = 0
        self.sentence_count = 0

    @staticmethod
    def _sentence_spans(text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans, cursor = [], 0
        for match in _SENTENCE.finditer(text):
            spans.append((cursor, match.start()))
            cursor = match.end()
        spans.append((cursor, len(text)))
        return spans

    def ingest(self, text: str, *, source: str = "memory", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = str(text or "")
        clean = normalize_text(raw)
        source_id = hashlib.sha256(f"{source}\0{clean}".encode("utf-8")).hexdigest()[:24]
        if source_id in self.documents:
            return {"source_id": source_id, "duplicate": True, "tokens": self.documents[source_id]["token_count"], "lexicon_size": len(self.lexemes)}
        spans = self._sentence_spans(clean)
        matches = list(_WORD.finditer(clean))
        tokens = [match.group(0).casefold() for match in matches]
        content_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        self.documents[source_id] = {"source_id": source_id, "source": source, "raw_text": raw, "normalized_text": clean, "sha256": content_hash, "token_count": len(tokens), "sentence_spans": spans, "metadata": dict(metadata or {})}
        self.document_terms[source_id] = Counter(tokens)
        self.total_tokens += len(tokens)
        self.sentence_count += len(spans)
        self.character_counts.update(clean)
        self.byte_counts.update(clean.encode("utf-8"))
        padded = f"^{clean.casefold()}$"
        for size in (2, 3, 4, 5):
            self.character_ngrams.update(padded[i:i + size] for i in range(max(0, len(padded) - size + 1)))
        for index, (token, match) in enumerate(zip(tokens, matches)):
            lexeme = self.lexemes.setdefault(token, Lexeme(token, stable_vector(token, self.dimensions)))
            lexeme.count += 1
            lexeme.documents.add(source_id)
            lexeme.forms[match.group(0)] += 1
            sentence_index = next((i for i, (start, end) in enumerate(spans) if start <= match.start() < end), max(0, len(spans) - 1))
            lexeme.occurrences.append(Occurrence(source_id, index, sentence_index, match.start(), match.end(), match.group(0)))
            for neighbor_index in range(max(0, index - self.window), min(len(tokens), index + self.window + 1)):
                if neighbor_index != index and tokens[neighbor_index] != token:
                    lexeme.neighbors[tokens[neighbor_index]] += 1.0 / abs(neighbor_index - index)
        for size in self.phrase_sizes:
            self.phrase_counts.update(" ".join(tokens[i:i + size]) for i in range(max(0, len(tokens) - size + 1)))
        return {"source_id": source_id, "duplicate": False, "tokens": len(tokens), "unique_tokens": len(set(tokens)), "sentences": len(spans), "characters": len(clean), "utf8_bytes": len(clean.encode("utf-8")), "lexicon_size": len(self.lexemes), "sha256": content_hash}

    def retrieve(self, query: str, *, top_k: int = 8) -> list[dict[str, Any]]:
        terms = words(query)
        if not terms:
            return []
        vectors = [stable_vector(term, self.dimensions) for term in terms]
        query_vector = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(self.dimensions))
        ranked = []
        for source_id, counts in self.document_terms.items():
            overlap = sum(counts[term] for term in set(terms))
            total = sum(counts.values()) or 1
            spatial = sum(max(0.0, cosine(query_vector, self.lexemes[token].vector)) * count / total for token, count in counts.most_common(128))
            score = overlap * 2.5 + spatial
            if score > 0:
                ranked.append((score, source_id))
        ranked.sort(reverse=True)
        return [{"source_id": source_id, "source": self.documents[source_id]["source"], "score": round(score, 6), "text": self.documents[source_id]["normalized_text"], "sha256": self.documents[source_id]["sha256"]} for score, source_id in ranked[:top_k]]

    def grab(self, query: str, *, top_k: int = 8, radius: int = 180) -> list[dict[str, Any]]:
        query_terms = set(words(query))
        rows = []
        for term in query_terms:
            for occurrence in (self.lexemes.get(term) or Lexeme(term, stable_vector(term))).occurrences:
                document = self.documents[occurrence.source_id]
                left = max(0, occurrence.start - radius)
                right = min(len(document["normalized_text"]), occurrence.end + radius)
                snippet = document["normalized_text"][left:right]
                rows.append({"source_id": occurrence.source_id, "source": document["source"], "term": term, "start": occurrence.start, "end": occurrence.end, "sentence_index": occurrence.sentence_index, "snippet": snippet, "score": len(query_terms & set(words(snippet))), "sha256": document["sha256"]})
        rows.sort(key=lambda row: (row["score"], row["source_id"], -row["start"]), reverse=True)
        output, seen = [], set()
        for row in rows:
            key = (row["source_id"], row["snippet"])
            if key not in seen:
                output.append(row)
                seen.add(key)
            if len(output) >= top_k:
                break
        return output

    def associations(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        terms = words(query)
        if not terms:
            return []
        vectors = [stable_vector(term, self.dimensions) for term in terms]
        query_vector = tuple(sum(vector[i] for vector in vectors) / len(vectors) for i in range(self.dimensions))
        direct: Counter[str] = Counter()
        for term in terms:
            if term in self.lexemes:
                direct.update(self.lexemes[term].neighbors)
        ranked = []
        for token, lexeme in self.lexemes.items():
            if token not in terms:
                score = cosine(query_vector, lexeme.vector) + math.log1p(direct[token]) * 0.45 + math.log1p(lexeme.count) * 0.08
                ranked.append((score, token, direct[token]))
        ranked.sort(reverse=True)
        return [{"token": token, "score": round(score, 6), "cooccurrence": round(float(cooccurrence), 6)} for score, token, cooccurrence in ranked[:top_k]]

    def manifest(self) -> dict[str, Any]:
        material = {"documents": [(key, value["sha256"], value["token_count"]) for key, value in sorted(self.documents.items())], "lexemes": [(key, value.count, sorted(value.documents), len(value.occurrences)) for key, value in sorted(self.lexemes.items())], "characters": sorted(self.character_counts.items()), "bytes": sorted(self.byte_counts.items()), "phrases": sorted(self.phrase_counts.items())}
        return {"schema": self.schema, "dimensions": self.dimensions, "window": self.window, "phrase_sizes": list(self.phrase_sizes), "documents": len(self.documents), "sentences": self.sentence_count, "tokens": self.total_tokens, "lexemes": len(self.lexemes), "characters": sum(self.character_counts.values()), "unique_characters": len(self.character_counts), "utf8_bytes": sum(self.byte_counts.values()), "character_ngrams": len(self.character_ngrams), "phrases": len(self.phrase_counts), "sha256": _hash(material)}

    def snapshot(self) -> dict[str, Any]:
        return {"schema": self.schema, "dimensions": self.dimensions, "window": self.window, "phrase_sizes": list(self.phrase_sizes), "documents": self.documents, "document_terms": {key: dict(value) for key, value in self.document_terms.items()}, "lexemes": {key: value.to_dict() for key, value in self.lexemes.items()}, "character_counts": dict(self.character_counts), "byte_counts": {str(key): value for key, value in self.byte_counts.items()}, "character_ngrams": dict(self.character_ngrams), "phrase_counts": dict(self.phrase_counts), "total_tokens": self.total_tokens, "sentence_count": self.sentence_count, "manifest": self.manifest()}

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "manifest": self.manifest()}

    @classmethod
    def load(cls, path: str | Path) -> "SpatialLexicon":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != cls.schema:
            raise ValueError(f"unsupported schema: {payload.get('schema')}")
        lexicon = cls(payload["dimensions"], payload["window"], payload["phrase_sizes"])
        lexicon.documents = payload["documents"]
        for value in lexicon.documents.values():
            value["sentence_spans"] = [tuple(span) for span in value["sentence_spans"]]
        lexicon.document_terms = {key: Counter(value) for key, value in payload["document_terms"].items()}
        for token, value in payload["lexemes"].items():
            lexicon.lexemes[token] = Lexeme(token, tuple(value["vector"]), value["count"], set(value["documents"]), Counter(value["neighbors"]), Counter(value["forms"]), [Occurrence(**item) for item in value["occurrences"]])
        lexicon.character_counts = Counter(payload["character_counts"])
        lexicon.byte_counts = Counter({int(key): value for key, value in payload["byte_counts"].items()})
        lexicon.character_ngrams = Counter(payload["character_ngrams"])
        lexicon.phrase_counts = Counter(payload["phrase_counts"])
        lexicon.total_tokens = payload["total_tokens"]
        lexicon.sentence_count = payload["sentence_count"]
        if lexicon.manifest()["sha256"] != payload["manifest"]["sha256"]:
            raise ValueError("lexicon manifest hash mismatch")
        return lexicon


class ExecutiveComposer:
    UNCERTAINTY = ("unknown", "cannot verify", "not inspected", "unconfirmed", "requires evidence", "not yet measured")

    def __init__(self, lexicon: SpatialLexicon | None = None) -> None:
        self.lexicon = lexicon or SpatialLexicon()

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", normalize_text(text)) if paragraph.strip()]

    @staticmethod
    def _signature(text: str) -> set[str]:
        return set(words(text))

    def deduplicate(self, text: str, threshold: float = 0.82) -> str:
        kept, signatures = [], []
        for paragraph in self._paragraphs(text):
            signature = self._signature(paragraph)
            if not any(len(signature & prior) / max(1, len(signature | prior)) >= threshold for prior in signatures):
                kept.append(paragraph)
                signatures.append(signature)
        return "\n\n".join(kept)

    def score(self, prompt: str, answer: str, evidence: Iterable[str] = ()) -> dict[str, float]:
        prompt_terms, answer_terms = set(words(prompt)), words(answer)
        answer_set = set(answer_terms)
        opening = set(words((_SENTENCE.split(answer, maxsplit=1) or [answer])[0]))
        evidence_terms = set(words("\n".join(evidence)))
        counts = Counter(answer_terms)
        repetition = sum(max(0, count - 3) for count in counts.values()) / max(1, len(answer_terms))
        return {"relevance": round(len(prompt_terms & answer_set) / max(1, len(prompt_terms)), 4), "directness": round(len(prompt_terms & opening) / max(1, min(len(prompt_terms), 16)), 4), "evidence_overlap": round(len(answer_set & evidence_terms) / max(1, len(answer_set)), 4) if evidence_terms else 0.0, "repetition": round(repetition, 4), "uncertainty_marked": float(any(marker in answer.casefold() for marker in self.UNCERTAINTY))}

    def compose(self, prompt: str, candidates: Iterable[str], *, evidence: Iterable[str] = (), require_uncertainty: bool = False, max_paragraphs: int = 8) -> dict[str, Any]:
        candidates = [normalize_text(item) for item in candidates if normalize_text(item)]
        evidence = [normalize_text(item) for item in evidence if normalize_text(item)]
        for index, item in enumerate(candidates):
            self.lexicon.ingest(item, source=f"candidate:{index}")
        for index, item in enumerate(evidence):
            self.lexicon.ingest(item, source=f"evidence:{index}")
        ranked = []
        for candidate_index, candidate in enumerate(candidates):
            for paragraph_index, paragraph in enumerate(self._paragraphs(candidate)):
                metrics = self.score(prompt, paragraph, evidence)
                aggregate = metrics["relevance"] * 0.45 + metrics["directness"] * 0.30 + metrics["evidence_overlap"] * 0.20 - metrics["repetition"] * 0.35
                ranked.append((aggregate, candidate_index, paragraph_index, paragraph, metrics))
        ranked.sort(reverse=True)
        selected, signatures = [], []
        for row in ranked:
            signature = self._signature(row[3])
            if any(len(signature & prior) / max(1, len(signature | prior)) >= 0.82 for prior in signatures):
                continue
            selected.append(row)
            signatures.append(signature)
            if len(selected) >= max_paragraphs:
                break
        if require_uncertainty:
            selected.sort(key=lambda row: (not any(marker in row[3].casefold() for marker in self.UNCERTAINTY), -row[0], row[1], row[2]))
        else:
            selected.sort(key=lambda row: (-row[0], row[1], row[2]))
        text = self.deduplicate("\n\n".join(row[3] for row in selected)) if selected else "I do not yet have enough grounded material to answer this directly."
        if require_uncertainty and not any(marker in text.casefold() for marker in self.UNCERTAINTY):
            text = "I cannot verify the strongest version of this claim from the evidence currently available.\n\n" + text
        return {"schema": "auro.executive-composition.v2", "text": text, "score": self.score(prompt, text, evidence), "candidate_count": len(candidates), "evidence_count": len(evidence), "selected_paragraphs": [{"aggregate": round(row[0], 6), "candidate_index": row[1], "paragraph_index": row[2], "metrics": row[4], "sha256": hashlib.sha256(row[3].encode("utf-8")).hexdigest()} for row in selected], "associations": self.lexicon.associations(prompt, top_k=16), "grabs": self.lexicon.grab(prompt, top_k=8), "lexicon": self.lexicon.manifest(), "claim_boundary": "This support layer does not establish frontier-model parity."}

    def creative_scaffold(self, prompt: str, *, branches: int = 8) -> dict[str, Any]:
        associations = self.lexicon.associations(prompt, top_k=max(24, branches * 3))
        terms = [row["token"] for row in associations]
        lenses = ("mechanism", "counterfactual", "analogy", "constraint", "failure", "synthesis", "scale-shift", "boundary inversion")
        rows = []
        for index in range(branches):
            related = terms[index * 3:index * 3 + 3] or terms[:3]
            lens = lenses[index % len(lenses)]
            rows.append({"branch": index + 1, "lens": lens, "related_terms": related, "instruction": f"Explore {prompt!r} through {lens} using {', '.join(related) or 'first principles'}; preserve contradictions and state falsification evidence."})
        return {"schema": "auro.creative-scaffold.v2", "prompt": prompt, "branches": rows, "lexicon": self.lexicon.manifest()}


class LanguageEngine:
    def __init__(self) -> None:
        self.lexicon = SpatialLexicon()
        self.composer = ExecutiveComposer(self.lexicon)
        self.last_receipt: dict[str, Any] | None = None

    def observe(self, text: str, *, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.lexicon.ingest(text, source=source, metadata=metadata)

    def grab(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        self.last_receipt = {"schema": "auro.language-grab.v1", "query": query, "documents": self.lexicon.retrieve(query, top_k=top_k), "spans": self.lexicon.grab(query, top_k=top_k), "associations": self.lexicon.associations(query, top_k=top_k * 2), "lexicon": self.lexicon.manifest()}
        return self.last_receipt

    def plan(self, prompt: str, *, branches: int = 8) -> dict[str, Any]:
        self.last_receipt = self.composer.creative_scaffold(prompt, branches=branches)
        return self.last_receipt

    def compose(self, prompt: str, candidates: Iterable[str], *, evidence: Iterable[str] = (), require_uncertainty: bool = False) -> dict[str, Any]:
        self.last_receipt = self.composer.compose(prompt, candidates, evidence=evidence, require_uncertainty=require_uncertainty)
        return self.last_receipt

    def revise(self, prompt: str, draft: str, critiques: Iterable[str], *, evidence: Iterable[str] = (), require_uncertainty: bool = False) -> dict[str, Any]:
        critique_list = [normalize_text(item) for item in critiques if normalize_text(item)]
        receipt = self.compose(prompt, [draft], evidence=[*evidence, *critique_list], require_uncertainty=require_uncertainty)
        receipt["schema"] = "auro.language-revision.v1"
        receipt["draft_sha256"] = hashlib.sha256(normalize_text(draft).encode("utf-8")).hexdigest()
        receipt["critique_count"] = len(critique_list)
        return receipt

    def save(self, path: str | Path) -> dict[str, Any]:
        return self.lexicon.save(path)

    def manifest(self) -> dict[str, Any]:
        return {"schema": "auro.language-engine.v1", "lexicon": self.lexicon.manifest(), "last_receipt_schema": (self.last_receipt or {}).get("schema")}
