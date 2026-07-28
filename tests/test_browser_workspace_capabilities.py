from pathlib import Path
from zipfile import ZipFile

from auro_native_llm.production_fleet.extensions import package_extension
from auro_native_llm.production_fleet.security import scan_workspace


def test_security_scanner_finds_secret_shape_without_leaking_value(tmp_path: Path) -> None:
    (tmp_path / "settings.txt").write_text('api_key="super-secret-value"', encoding="utf-8")
    result = scan_workspace(tmp_path)
    assert result["files_scanned"] == 1
    assert result["findings"][0]["kind"] == "generic_secret"
    assert "super-secret-value" not in str(result)


def test_extension_package_is_installable_manifest_v3_zip(tmp_path: Path) -> None:
    result = package_extension(tmp_path)
    archive = Path(result["path"])
    with ZipFile(archive) as package:
        assert {"manifest.json", "popup.html", "popup.js"} <= set(package.namelist())
        assert '"manifest_version": 3' in package.read("manifest.json").decode()
    assert len(result["sha256"]) == 64
