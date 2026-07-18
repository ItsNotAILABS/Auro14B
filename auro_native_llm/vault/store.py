"""Multi-ledger sealed vault for HIM high-value secrets.

Ledgers
-------
  keys       — signing / API material references
  rpc        — Alchemy/Infura and chain RPC URLs
  high_value — tokens, wallet mnemonics refs, high-value blobs
  agent      — agent session tokens, mint receipts
  github     — gh PAT refs (prefer OS keyring; store only if needed)

Encryption
----------
  1) Windows DPAPI when available (user-scope)
  2) Password + PBKDF2-HMAC-SHA256 + XOR stream + HMAC (stdlib portable)
     Not as strong as AES-GCM; upgrade with cryptography when installed.

Never logs secret plaintext. list() returns metadata only.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGERS = ("keys", "rpc", "high_value", "agent", "github")
_DEFAULT_ROOT = Path.home() / ".auro_vault"
_VAULT: Optional["Vault"] = None


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _dpapi_protect(data: bytes) -> Optional[bytes]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32

        blob_in = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
        blob_out = DATA_BLOB()
        if not crypt32.CryptProtectData(
            ctypes.byref(blob_in),
            "HIM-Vault",
            None,
            None,
            None,
            0,
            ctypes.byref(blob_out),
        ):
            return None
        try:
            out = ctypes.string_at(blob_out.pbData, blob_out.cbData)
            return bytes(out)
        finally:
            kernel32.LocalFree(blob_out.pbData)
    except Exception:
        return None


def _dpapi_unprotect(data: bytes) -> Optional[bytes]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class DATA_BLOB(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        blob_in = DATA_BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
        blob_out = DATA_BLOB()
        if not crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(blob_out),
        ):
            return None
        try:
            return bytes(ctypes.string_at(blob_out.pbData, blob_out.cbData))
        finally:
            kernel32.LocalFree(blob_out.pbData)
    except Exception:
        return None


def _derive_key(password: str, salt: bytes, rounds: int = 200_000) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds, dklen=32)


def _stream_crypt(data: bytes, key: bytes, nonce: bytes) -> bytes:
    """HMAC-DRBG style keystream XOR (stdlib portable)."""
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


@dataclass
class VaultEntryMeta:
    entry_id: str
    ledger: str
    name: str
    created_at: float
    updated_at: float
    backend: str
    sha256_prefix: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "ledger": self.ledger,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "backend": self.backend,
            "sha256_prefix": self.sha256_prefix,
        }


class Vault:
    def __init__(
        self,
        root: Optional[Path] = None,
        *,
        password: Optional[str] = None,
    ) -> None:
        self.root = Path(root or os.environ.get("AURO_VAULT_ROOT") or _DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)
        self.password = password or os.environ.get("AURO_VAULT_PASSWORD") or ""
        self._index_path = self.root / "index.json"
        self._index: Dict[str, Any] = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"schema": "auro.vault.v1", "ledgers": {k: {} for k in LEDGERS}, "version": 1}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")

    def _ledger_path(self, ledger: str, entry_id: str) -> Path:
        d = self.root / ledger
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{entry_id}.sealed"

    def _seal(self, plaintext: str) -> Dict[str, Any]:
        raw = plaintext.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        # Prefer DPAPI on Windows when no password forced
        if not self.password:
            prot = _dpapi_protect(raw)
            if prot is not None:
                return {
                    "backend": "dpapi",
                    "blob": _b64e(prot),
                    "sha256": digest,
                }
        # Password path (or DPAPI unavailable)
        pwd = self.password or "him-local-dev-only-change-me"
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(16)
        key = _derive_key(pwd, salt)
        ct = _stream_crypt(raw, key, nonce)
        tag = hmac.new(key, nonce + ct, hashlib.sha256).digest()
        return {
            "backend": "pbkdf2_hmac_stream",
            "salt": _b64e(salt),
            "nonce": _b64e(nonce),
            "blob": _b64e(ct),
            "tag": _b64e(tag),
            "sha256": digest,
            "rounds": 200_000,
        }

    def _unseal(self, sealed: Dict[str, Any]) -> str:
        backend = sealed.get("backend")
        if backend == "dpapi":
            raw = _dpapi_unprotect(_b64d(sealed["blob"]))
            if raw is None:
                raise RuntimeError("DPAPI unprotect failed")
            return raw.decode("utf-8")
        if backend == "pbkdf2_hmac_stream":
            pwd = self.password or "him-local-dev-only-change-me"
            salt = _b64d(sealed["salt"])
            nonce = _b64d(sealed["nonce"])
            ct = _b64d(sealed["blob"])
            tag = _b64d(sealed["tag"])
            key = _derive_key(pwd, salt, int(sealed.get("rounds") or 200_000))
            expect = hmac.new(key, nonce + ct, hashlib.sha256).digest()
            if not hmac.compare_digest(tag, expect):
                raise RuntimeError("vault HMAC verification failed")
            pt = _stream_crypt(ct, key, nonce)
            return pt.decode("utf-8")
        raise RuntimeError(f"unknown vault backend {backend}")

    def put(
        self,
        ledger: str,
        name: str,
        secret: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if ledger not in LEDGERS:
            return {"ok": False, "error": f"unknown ledger {ledger}", "ledgers": list(LEDGERS)}
        if not name or secret is None:
            return {"ok": False, "error": "name and secret required"}
        entry_id = f"e-{uuid.uuid4().hex[:12]}"
        now = time.time()
        sealed = self._seal(str(secret))
        path = self._ledger_path(ledger, entry_id)
        payload = {
            "entry_id": entry_id,
            "ledger": ledger,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "meta": meta or {},
            "sealed": sealed,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        self._index["ledgers"][ledger][entry_id] = {
            "name": name,
            "created_at": now,
            "updated_at": now,
            "backend": sealed["backend"],
            "sha256_prefix": sealed["sha256"][:12],
            "meta": meta or {},
        }
        self._save_index()
        return {
            "ok": True,
            "entry_id": entry_id,
            "ledger": ledger,
            "name": name,
            "backend": sealed["backend"],
            "sha256_prefix": sealed["sha256"][:12],
            # never return secret
        }

    def get(self, ledger: str, entry_id: str, *, reveal: bool = False) -> Dict[str, Any]:
        if ledger not in LEDGERS:
            return {"ok": False, "error": f"unknown ledger {ledger}"}
        path = self._ledger_path(ledger, entry_id)
        if not path.exists():
            return {"ok": False, "error": "not found"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        out = {
            "ok": True,
            "entry_id": entry_id,
            "ledger": ledger,
            "name": payload.get("name"),
            "backend": (payload.get("sealed") or {}).get("backend"),
            "meta": payload.get("meta") or {},
            "revealed": False,
        }
        if reveal:
            try:
                out["secret"] = self._unseal(payload["sealed"])
                out["revealed"] = True
            except Exception as exc:
                return {"ok": False, "error": str(exc)[:200]}
        return out

    def get_by_name(self, ledger: str, name: str, *, reveal: bool = False) -> Dict[str, Any]:
        for eid, meta in (self._index.get("ledgers") or {}).get(ledger, {}).items():
            if meta.get("name") == name:
                return self.get(ledger, eid, reveal=reveal)
        return {"ok": False, "error": f"name {name!r} not in {ledger}"}

    def list(self, ledger: Optional[str] = None) -> Dict[str, Any]:
        ledgers = [ledger] if ledger else list(LEDGERS)
        out: Dict[str, Any] = {"ok": True, "ledgers": {}}
        for lg in ledgers:
            if lg not in LEDGERS:
                continue
            entries = []
            for eid, meta in (self._index.get("ledgers") or {}).get(lg, {}).items():
                entries.append({"entry_id": eid, **meta})
            out["ledgers"][lg] = entries
        return out

    def delete(self, ledger: str, entry_id: str) -> Dict[str, Any]:
        path = self._ledger_path(ledger, entry_id)
        if path.exists():
            path.unlink()
        (self._index.get("ledgers") or {}).get(ledger, {}).pop(entry_id, None)
        self._save_index()
        return {"ok": True, "deleted": entry_id}

    def export_rpc_env_hint(self) -> Dict[str, Any]:
        """Metadata-only hint for wiring him-web3 without dumping secrets."""
        rpc_entries = (self._index.get("ledgers") or {}).get("rpc") or {}
        return {
            "ok": True,
            "n_rpc_secrets": len(rpc_entries),
            "names": [m.get("name") for m in rpc_entries.values()],
            "hint": "Use vault.get_by_name('rpc', name, reveal=True) only in server process; write to him-web3/.env locally.",
        }

    def health(self) -> Dict[str, Any]:
        counts = {lg: len((self._index.get("ledgers") or {}).get(lg) or {}) for lg in LEDGERS}
        return {
            "schema": "auro.vault.health.v1",
            "root": str(self.root),
            "ledgers": list(LEDGERS),
            "counts": counts,
            "total": sum(counts.values()),
            "password_set": bool(self.password),
            "dpapi_platform": os.name == "nt",
        }


def get_vault(password: Optional[str] = None, root: Optional[Path] = None) -> Vault:
    global _VAULT
    if _VAULT is None or password is not None or root is not None:
        _VAULT = Vault(root=root, password=password)
    return _VAULT
