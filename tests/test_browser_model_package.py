from pathlib import Path

import pytest

from scripts.prepare_browser_model import prepare


def test_browser_model_manifest_requires_real_onnx(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No ONNX"):
        prepare(tmp_path)


def test_browser_model_manifest_hashes_export(tmp_path: Path) -> None:
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"real-export-placeholder-for-test")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    manifest = prepare(tmp_path)
    assert manifest["onnx_files"] == 1
    assert len(manifest["manifest_sha256"]) == 64
    assert (tmp_path / "auro-browser-manifest.json").is_file()
