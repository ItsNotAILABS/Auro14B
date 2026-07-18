"""Bounded static workspace scanner; not a malware sandbox."""
from __future__ import annotations
import hashlib, re
from pathlib import Path

SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b")),
    ("generic_secret", re.compile(r"(?i)\b(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{8,}")),
)
RISKY_SUFFIXES = {".exe", ".dll", ".ps1", ".bat", ".cmd", ".scr", ".msi"}
EXCLUDED = {".git", "node_modules", "dist", "target", "vendor"}


def scan_workspace(root, *, max_files=2000, max_file_bytes=1_048_576):
    base = Path(root).resolve()
    if not base.is_dir(): raise ValueError(f"workspace is not a directory: {base}")
    findings=[]; files=[]; skipped=0
    for path in sorted(base.rglob("*")):
        if len(files) >= max(1, min(int(max_files), 10000)): break
        if not path.is_file() or any(part in EXCLUDED for part in path.parts): continue
        relative=path.relative_to(base).as_posix(); size=path.stat().st_size
        if size > max_file_bytes: skipped+=1; continue
        data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        files.append({"path":relative,"bytes":size,"sha256":digest})
        if path.suffix.lower() in RISKY_SUFFIXES:
            findings.append({"severity":"review","kind":"executable_file","path":relative})
        if b"\x00" not in data[:4096]:
            text=data.decode("utf-8",errors="ignore")
            for kind,pattern in SECRET_PATTERNS:
                if pattern.search(text): findings.append({"severity":"high","kind":kind,"path":relative})
    payload={"schema":"auro.security.scan.v1","root":str(base),"files_scanned":len(files),"files_skipped":skipped,"findings":findings,"static_only":True,"malware_sandbox":False}
    payload["scan_sha256"]=hashlib.sha256(str([(x["path"],x["sha256"]) for x in files]).encode()).hexdigest()
    return payload
