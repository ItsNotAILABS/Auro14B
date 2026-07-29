"""Deterministic structured-report to conversational-text fluidizer.

The module is pure Python and Pyodide-compatible. It adds no facts: it only
orders, deduplicates and grammatically connects text already present in a
structured AURO report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FluidizedResult:
    text: str
    source_sha256: str
    output_sha256: str
    sentences_in: int
    sentences_out: int
    dropped_duplicates: int
    schema: str = "auro.python_wasm.fluid_text.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def _sentences(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        blocks = re.split(r"(?<=[.!?])\s+|\n+|\s*[•*-]\s+", value)
        return [" ".join(item.split()) for item in blocks if item and item.strip()]
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            if key in {"raw", "receipt", "metadata", "actions"}:
                continue
            out.extend(_sentences(item))
        return out
    if isinstance(value, Iterable):
        out = []
        for item in value:
            out.extend(_sentences(item))
        return out
    return [str(value)]


def _normalize(sentence: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()


def _finish(sentence: str) -> str:
    text = sentence.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:] if text[0].isalpha() else text
    return text if text[-1] in ".!?;:)`]" else text + "."


def fluidize_report(report: Mapping[str, Any] | str, *, voice: str = "direct", max_sentences: int = 18) -> FluidizedResult:
    source: Any = report
    if isinstance(report, str):
        try:
            parsed = json.loads(report)
            source = parsed if isinstance(parsed, Mapping) else {"answer": report}
        except json.JSONDecodeError:
            source = {"answer": report}
    if not isinstance(source, Mapping):
        source = {"answer": str(source)}

    ordered_keys = (
        "answer",
        "consensus",
        "summary",
        "key_points",
        "recommendations",
        "reasoning_summary",
        "caveats",
        "limitations",
        "next_steps",
    )
    raw: list[str] = []
    for key in ordered_keys:
        raw.extend(_sentences(source.get(key)))
    if not raw:
        raw.extend(_sentences(source))

    unique: list[str] = []
    seen: set[str] = set()
    dropped = 0
    for sentence in raw:
        key = _normalize(sentence)
        if not key or key in seen:
            dropped += int(bool(key))
            continue
        seen.add(key)
        unique.append(_finish(sentence))
        if len(unique) >= max(1, int(max_sentences)):
            break

    if voice == "concise":
        unique = unique[:8]
    elif voice == "technical":
        pass
    elif voice not in {"direct", "conversational"}:
        raise ValueError("voice must be direct, conversational, concise, or technical")

    if not unique:
        text = "No supported response content was produced."
    elif len(unique) == 1:
        text = unique[0]
    else:
        transitions = ["", "More specifically, ", "At the same time, ", "Operationally, ", "The key boundary is that "]
        rendered = [unique[0]]
        for index, sentence in enumerate(unique[1:], 1):
            if voice == "technical" or sentence.startswith(("```", "http")):
                rendered.append(sentence)
            else:
                prefix = transitions[min(index, len(transitions) - 1)]
                rendered.append(prefix + sentence[0].lower() + sentence[1:] if prefix and sentence else sentence)
        text = " ".join(rendered)

    citations = source.get("citations")
    citation_values = [str(item) for item in citations] if isinstance(citations, list) else []
    if citation_values:
        text += " Sources: " + "; ".join(dict.fromkeys(citation_values)) + "."

    source_hash = hashlib.sha256(_canonical(source)).hexdigest()
    output_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return FluidizedResult(
        text=text,
        source_sha256=source_hash,
        output_sha256=output_hash,
        sentences_in=len(raw),
        sentences_out=len(unique),
        dropped_duplicates=dropped,
    )
