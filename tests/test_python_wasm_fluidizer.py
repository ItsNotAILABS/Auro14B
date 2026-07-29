from pathlib import Path

from auro_native_llm.model.fluidizer import fluidize_report


def test_fluidizer_orders_deduplicates_and_preserves_source_terms():
    report = {
        "answer": "The model should use bounded task capsules.",
        "key_points": ["The model should use bounded task capsules.", "MESIE runs at every stage."],
        "caveats": ["This fixture is not benchmark evidence."],
        "next_steps": ["Run the exact checkpoints."],
        "citations": ["https://example.test/source"],
    }
    first = fluidize_report(report, voice="conversational")
    second = fluidize_report(report, voice="conversational")
    assert first.text == second.text
    assert first.source_sha256 == second.source_sha256
    assert first.output_sha256 == second.output_sha256
    assert first.text.count("bounded task capsules") == 1
    for phrase in ("MESIE", "not benchmark evidence", "exact checkpoints", "https://example.test/source"):
        assert phrase in first.text
    assert "beats" not in first.text.lower()
    assert first.dropped_duplicates >= 1


def test_browser_fluidizer_is_local_pyodide_and_disallows_remote_packages():
    source = Path("browser-brain/src/python-wasm-fluidizer.js").read_text(encoding="utf-8")
    assert "remotePackagesAllowed:false" in source
    assert "loadPyodide" in source
    assert "result_json" in source
    assert "micropip" not in source
