"""Capability discovery sequence — actual environment, not stereotypes.

Sequence (mandatory before inability claims):
  1. Inspect the repository
  2. Read project manifests
  3. Inspect installed tools and runtimes
  4. Check existing build configuration
  5. Search for prior successful implementations
  6. Run a harmless capability probe
  7. Identify the actual source of any failure
  8. Only then report a constraint (classified)
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from auro_native_llm.capability.taxonomy import (
    CapabilityStatus,
    STATUS_MEANING,
    cannot_collapse,
    classify_message,
)

# Repo root = package parent of auro_native_llm
_REPO = Path(__file__).resolve().parents[2]


@dataclass
class CapEntry:
    name: str
    status: CapabilityStatus
    evidence: str
    step: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "evidence": self.evidence,
            "step": self.step,
            "detail": self.detail,
            "meaning": STATUS_MEANING[self.status],
        }


@dataclass
class CapabilityReport:
    ok: bool
    repo_root: str
    steps: Dict[str, Any]
    capabilities: List[CapEntry]
    elapsed_s: float
    claim: str = (
        "Discovery precedes inability. Status is one of eight classes — never a bare 'I can't'."
    )

    def by_status(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {s.value: [] for s in CapabilityStatus}
        for c in self.capabilities:
            out[c.status.value].append(c.name)
        return out

    def available(self) -> List[str]:
        return [
            c.name
            for c in self.capabilities
            if c.status
            in (CapabilityStatus.AVAILABLE, CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED)
        ]

    def constraints(self) -> List[Dict[str, Any]]:
        """Only non-available, fully classified."""
        return [
            c.to_dict()
            for c in self.capabilities
            if c.status
            not in (
                CapabilityStatus.AVAILABLE,
                CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED,
            )
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "auro.capability.discovery.v1",
            "ok": self.ok,
            "repo_root": self.repo_root,
            "elapsed_s": self.elapsed_s,
            "claim": self.claim,
            "anti_pattern": "Never collapse all failures into 'I can't'.",
            "taxonomy": [s.value for s in CapabilityStatus],
            "by_status": self.by_status(),
            "available": self.available(),
            "constraints": self.constraints(),
            "n_capabilities": len(self.capabilities),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "steps": self.steps,
        }

    def assert_or_classify(self, feature: str) -> str:
        for c in self.capabilities:
            if c.name == feature or feature in c.name:
                if c.status in (
                    CapabilityStatus.AVAILABLE,
                    CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED,
                ):
                    return f"[{c.status.value}] {feature}: {c.evidence}"
                return cannot_collapse(c.status, feature, c.evidence)
        return cannot_collapse(
            CapabilityStatus.UNKNOWN,
            feature,
            "not present in last discovery report — re-run discovery",
        )


def _run(cmd: List[str], timeout: float = 8.0) -> Tuple[int, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out.strip()[:500]
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:
        return 1, str(exc)[:200]


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _add(
    caps: List[CapEntry],
    name: str,
    status: CapabilityStatus,
    evidence: str,
    step: int,
    **detail: Any,
) -> None:
    caps.append(CapEntry(name=name, status=status, evidence=evidence, step=step, detail=detail))


def discover_capabilities(
    repo_root: Optional[Path] = None,
    *,
    probes: bool = True,
) -> CapabilityReport:
    """Full 8-step capability-state discovery against the *actual* environment."""
    t0 = time.time()
    root = Path(repo_root or _REPO).resolve()
    caps: List[CapEntry] = []
    steps: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 1. Repo
    step = 1
    repo_files = []
    if root.exists():
        try:
            repo_files = sorted(
                p.name for p in root.iterdir() if not p.name.startswith(".")
            )[:80]
        except Exception as exc:
            _add(caps, "repo.list", CapabilityStatus.BLOCKED_BY_PERMISSION, str(exc), step)
        else:
            _add(
                caps,
                "repo.root",
                CapabilityStatus.AVAILABLE,
                str(root),
                step,
                entries=len(repo_files),
            )
        for name in (
            "auro_native_llm",
            "mesie",
            "checkpoints",
            "bindings",
            "scripts",
            "tests",
            "artifacts",
            "phantom_native",
            "auro_foundry",
        ):
            p = root / name
            if p.exists():
                _add(caps, f"repo.{name}", CapabilityStatus.AVAILABLE, str(p), step)
            else:
                _add(
                    caps,
                    f"repo.{name}",
                    CapabilityStatus.TECHNICALLY_UNAVAILABLE,
                    f"missing under {root}",
                    step,
                )
    else:
        _add(caps, "repo.root", CapabilityStatus.TECHNICALLY_UNAVAILABLE, str(root), step)
    steps["1_inspect_repository"] = {"root": str(root), "top_level": repo_files[:40]}

    # ------------------------------------------------------------------ 2. Manifests
    step = 2
    manifests = {
        "pyproject.toml": root / "pyproject.toml",
        "requirements.txt": root / "requirements.txt",
        "package.json": root / "package.json",
        "Cargo.toml": root / "bindings" / "rust" / "Cargo.toml",
        "Project.toml": root / "bindings" / "julia" / "AuroCompute" / "Project.toml",
        "AGENTS.md": root / "AGENTS.md",
        "README.md": root / "README.md",
    }
    # also search julia Project.toml variants
    for jp in (root / "bindings" / "julia").rglob("Project.toml") if (root / "bindings" / "julia").exists() else []:
        manifests[f"julia:{jp.parent.name}"] = jp
        break
    man_found = {}
    for label, path in manifests.items():
        if path.exists():
            man_found[label] = str(path)
            _add(caps, f"manifest.{label}", CapabilityStatus.AVAILABLE, str(path), step)
            # peek pyproject for package name
            if path.name == "pyproject.toml":
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                    _add(
                        caps,
                        "manifest.pyproject_readable",
                        CapabilityStatus.AVAILABLE,
                        f"{len(text)} chars",
                        step,
                    )
                except Exception as exc:
                    _add(
                        caps,
                        "manifest.pyproject_readable",
                        CapabilityStatus.BLOCKED_BY_PERMISSION,
                        str(exc),
                        step,
                    )
        else:
            # package.json missing is common for pure python monorepos — not "I can't python"
            st = (
                CapabilityStatus.UNSUPPORTED_BY_DEFAULT_TEMPLATE
                if label == "package.json"
                else CapabilityStatus.NOT_CURRENTLY_CONFIGURED
            )
            _add(caps, f"manifest.{label}", st, f"not at {path}", step)
    steps["2_read_manifests"] = man_found

    # ------------------------------------------------------------------ 3. Tools / runtimes
    step = 3
    tools = {
        "python": [sys.executable, "--version"],
        "git": ["git", "--version"],
        "gh": ["gh", "--version"],
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
        "julia": ["julia", "--version"],
        "cargo": ["cargo", "--version"],
        "rustc": ["rustc", "--version"],
        "ghc": ["ghc", "--version"],
        "zig": ["zig", "version"],
    }
    runtime_info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }
    for name, cmd in tools.items():
        path = _which(cmd[0]) if cmd[0] != sys.executable else sys.executable
        if path or (name == "python"):
            code, out = _run(cmd if name != "python" else [sys.executable, "--version"])
            if code == 0 or name == "python":
                _add(
                    caps,
                    f"runtime.{name}",
                    CapabilityStatus.AVAILABLE,
                    out or path or "ok",
                    step,
                    path=path,
                )
                runtime_info[name] = out or path
            else:
                # binary on PATH but failed
                _add(
                    caps,
                    f"runtime.{name}",
                    CapabilityStatus.NOT_CURRENTLY_CONFIGURED,
                    out or f"exit {code}",
                    step,
                    path=path,
                )
        else:
            # special: ghc often absent on Windows ARM — technical
            st = CapabilityStatus.TECHNICALLY_UNAVAILABLE
            _add(caps, f"runtime.{name}", st, "not on PATH", step)
    steps["3_tools_runtimes"] = runtime_info

    # Python packages
    pkg_names = [
        "numpy",
        "mesie",
        "torch",
        "scipy",
        "pandas",
        "sklearn",
        "networkx",
    ]
    for pkg in pkg_names:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            ver = getattr(mod, "__version__", "?")
            path = getattr(mod, "__file__", "?")
            _add(
                caps,
                f"pip.{pkg}",
                CapabilityStatus.AVAILABLE,
                f"v{ver} @ {path}",
                step,
                version=ver,
            )
        except Exception as exc:
            # torch often missing on Win ARM — technical, not "can't do ML"
            st = (
                CapabilityStatus.TECHNICALLY_UNAVAILABLE
                if pkg == "torch"
                else CapabilityStatus.NOT_CURRENTLY_CONFIGURED
            )
            _add(caps, f"pip.{pkg}", st, str(exc)[:160], step)

    # ------------------------------------------------------------------ 4. Build config
    step = 4
    build_bits = {
        "pyproject_scripts": False,
        "tests_dir": (root / "tests").is_dir(),
        "checkpoints": (root / "checkpoints" / "auro_minds").is_dir(),
        "native_llm_configs": (root / "native_llm" / "configs").is_dir(),
        "dockerfile": (root / "runner" / "Dockerfile.cuda").exists(),
        "github_workflow": (root / ".github" / "workflows").is_dir(),
    }
    # parse pyproject for scripts/entry points
    pp = root / "pyproject.toml"
    if pp.exists():
        text = pp.read_text(encoding="utf-8", errors="ignore")
        build_bits["pyproject_scripts"] = "[project.scripts]" in text or "scripts" in text
        _add(caps, "build.pyproject", CapabilityStatus.AVAILABLE, "present", step)
    if build_bits["tests_dir"]:
        n_tests = len(list((root / "tests").rglob("test_*.py")))
        _add(caps, "build.tests", CapabilityStatus.AVAILABLE, f"{n_tests} test_*.py", step, n=n_tests)
    else:
        _add(caps, "build.tests", CapabilityStatus.NOT_CURRENTLY_CONFIGURED, "no tests/", step)
    if build_bits["checkpoints"]:
        ck = list((root / "checkpoints" / "auro_minds").iterdir())
        names = [p.name for p in ck if p.is_dir()]
        _add(
            caps,
            "build.checkpoints",
            CapabilityStatus.AVAILABLE,
            f"{len(names)} mind ckpts: {', '.join(names[:8])}",
            step,
            names=names,
        )
    else:
        _add(caps, "build.checkpoints", CapabilityStatus.UNTESTED, "no checkpoints yet", step)
    if build_bits["dockerfile"]:
        _add(caps, "build.cuda_dockerfile", CapabilityStatus.AVAILABLE, "runner/Dockerfile.cuda", step)
    else:
        _add(
            caps,
            "build.cuda_dockerfile",
            CapabilityStatus.NOT_CURRENTLY_CONFIGURED,
            "missing",
            step,
        )
    steps["4_build_configuration"] = build_bits

    # ------------------------------------------------------------------ 5. Prior success
    step = 5
    prior = {}
    art = root / "artifacts"
    if art.exists():
        for sub in ("auro-readiness", "auro-ghost", "auro-long-harness"):
            p = art / sub
            if p.exists():
                files = list(p.glob("*"))
                prior[sub] = [f.name for f in files[:12]]
                _add(
                    caps,
                    f"prior.{sub}",
                    CapabilityStatus.AVAILABLE,
                    f"{len(files)} artifacts",
                    step,
                )
            else:
                _add(
                    caps,
                    f"prior.{sub}",
                    CapabilityStatus.UNTESTED,
                    "no artifact dir",
                    step,
                )
    # promotion receipt
    promo = root / "artifacts" / "auro-readiness" / "PROMOTION_RECEIPT.json"
    if promo.exists():
        try:
            data = json.loads(promo.read_text(encoding="utf-8"))
            _add(
                caps,
                "prior.promotion_receipt",
                CapabilityStatus.AVAILABLE,
                f"tier={data.get('tier') or (data.get('readiness') or {}).get('tier')}",
                step,
                keys=list(data.keys())[:12],
            )
        except Exception as exc:
            _add(caps, "prior.promotion_receipt", CapabilityStatus.UNTESTED, str(exc), step)
    # physics train report
    phys = root / "checkpoints" / "auro_minds" / "Auro-2B_physics" / "PHYSICS_TRAIN_REPORT.json"
    if phys.exists():
        try:
            data = json.loads(phys.read_text(encoding="utf-8"))
            _add(
                caps,
                "prior.physics_train",
                CapabilityStatus.AVAILABLE,
                f"ce {data.get('ce_first')}→{data.get('ce_last')} scaffold={data.get('scaffold')}",
                step,
            )
        except Exception as exc:
            _add(caps, "prior.physics_train", CapabilityStatus.UNTESTED, str(exc), step)
    else:
        _add(caps, "prior.physics_train", CapabilityStatus.UNTESTED, "no physics ckpt report", step)
    # specialize
    spec = root / "checkpoints" / "auro_minds" / "Auro-2B_specialized" / "SPECIALIZE_REPORT.json"
    if spec.exists():
        _add(caps, "prior.specialize", CapabilityStatus.AVAILABLE, str(spec), step)
    steps["5_prior_success"] = prior

    # ------------------------------------------------------------------ 6. Harmless probes
    step = 6
    probe_results: Dict[str, Any] = {}
    if probes:
        # numpy matmul
        try:
            import numpy as np

            a = np.eye(4)
            b = a @ a
            assert b[0, 0] == 1.0
            _add(caps, "probe.numpy_matmul", CapabilityStatus.AVAILABLE, "4x4 eye@eye ok", step)
            probe_results["numpy"] = "ok"
        except Exception as exc:
            _add(caps, "probe.numpy_matmul", CapabilityStatus.TECHNICALLY_UNAVAILABLE, str(exc), step)

        # mesie import + SpectralGPT tiny forward
        try:
            import mesie

            ver = getattr(mesie, "__version__", "?")
            path = getattr(mesie, "__file__", "?")
            _add(caps, "probe.mesie_import", CapabilityStatus.AVAILABLE, f"v{ver}", step, path=path)
            from mesie.foundation import SpectralGPT

            m = SpectralGPT(
                hidden_dim=32,
                num_layers=1,
                num_heads=2,
                head_dim=16,
                vocab_size=64,
                max_seq_len=16,
                ffn_dim=64,
                use_moe=False,
                use_cross_modal=False,
                use_spectral_encoder=False,
                causal=True,
            )
            import numpy as np

            ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
            out = m.forward(ids) if hasattr(m, "forward") else m(ids)
            _add(
                caps,
                "probe.spectral_gpt_forward",
                CapabilityStatus.AVAILABLE,
                f"forward type={type(out).__name__}",
                step,
            )
            probe_results["mesie"] = ver
        except Exception as exc:
            _add(
                caps,
                "probe.mesie",
                CapabilityStatus.TECHNICALLY_UNAVAILABLE,
                str(exc)[:200],
                step,
            )

        # physics formulas
        try:
            from auro_native_llm.physics.formulas import dispersion_omega, PHI
            import numpy as np

            w = dispersion_omega(np.linspace(0, float(PHI), 8))
            assert w.size == 8 and float(w.min()) > 0
            _add(
                caps,
                "probe.physics_dispersion",
                CapabilityStatus.AVAILABLE,
                f"ω̄={float(w.mean()):.4f}",
                step,
            )
        except Exception as exc:
            _add(
                caps,
                "probe.physics",
                CapabilityStatus.NOT_CURRENTLY_CONFIGURED,
                str(exc)[:160],
                step,
            )

        # auro mind lite build (light — import only)
        try:
            from auro_native_llm.model.auro_lm import AuroLanguageModel
            from auro_native_llm.model.config import mesie_preset_dims

            tiny = mesie_preset_dims("spectral_gpt_tiny")
            _add(
                caps,
                "probe.auro_lm_config",
                CapabilityStatus.AVAILABLE,
                f"tiny dims keys={list(tiny)[:6]}",
                step,
            )
        except Exception as exc:
            _add(caps, "probe.auro_lm", CapabilityStatus.NOT_CURRENTLY_CONFIGURED, str(exc)[:160], step)

        # chrome path discovery (not launch)
        try:
            from auro_native_llm.chrome.cdp import ChromeCDP

            c = ChromeCDP(mock=True)
            path = c._chrome_path()
            if path and os.path.exists(path):
                _add(
                    caps,
                    "probe.chrome_binary",
                    CapabilityStatus.AVAILABLE,
                    path,
                    step,
                )
            else:
                _add(
                    caps,
                    "probe.chrome_binary",
                    CapabilityStatus.TECHNICALLY_UNAVAILABLE,
                    f"path={path}",
                    step,
                )
            _add(caps, "probe.chrome_mock", CapabilityStatus.AVAILABLE, "mock CDP ok", step)
        except Exception as exc:
            _add(caps, "probe.chrome", CapabilityStatus.UNTESTED, str(exc)[:160], step)

        # google envelope
        try:
            from auro_native_llm.gworkspace import get_envelope

            env = get_envelope(None, chrome_mock=True, force=True)
            h = env.act("search", query="capability discovery")
            _add(
                caps,
                "probe.google_envelope",
                CapabilityStatus.AVAILABLE if h.get("ok") else CapabilityStatus.UNTESTED,
                f"envelope={env.envelope_id}",
                step,
            )
        except Exception as exc:
            _add(caps, "probe.google_envelope", CapabilityStatus.NOT_CURRENTLY_CONFIGURED, str(exc)[:160], step)

        # ghost policy
        try:
            from auro_native_llm.ghost.policy import PolicyGate

            d = PolicyGate().decide("embed spectral match")
            _add(
                caps,
                "probe.ghost_policy",
                CapabilityStatus.AVAILABLE,
                f"risk={d.risk_class.name} allowed={d.allowed}",
                step,
            )
        except Exception as exc:
            _add(caps, "probe.ghost", CapabilityStatus.NOT_CURRENTLY_CONFIGURED, str(exc)[:160], step)

        # julia one-liner
        if _which("julia"):
            code, out = _run(["julia", "-e", "println(1+1)"], timeout=30.0)
            if code == 0 and "2" in out:
                _add(caps, "probe.julia_eval", CapabilityStatus.AVAILABLE, out, step)
            else:
                _add(
                    caps,
                    "probe.julia_eval",
                    CapabilityStatus.NOT_CURRENTLY_CONFIGURED,
                    out or f"exit {code}",
                    step,
                )
    steps["6_harmless_probes"] = probe_results

    # ------------------------------------------------------------------ 7. Failure sources
    step = 7
    failures = [c for c in caps if c.status not in (
        CapabilityStatus.AVAILABLE,
        CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED,
    )]
    sources: Dict[str, List[str]] = {}
    for c in failures:
        sources.setdefault(c.status.value, []).append(c.name)
    # classic stereotype trap: torch missing on Win ARM
    if any(c.name == "pip.torch" and c.status == CapabilityStatus.TECHNICALLY_UNAVAILABLE for c in caps):
        _add(
            caps,
            "diagnosis.torch_win_arm",
            CapabilityStatus.TECHNICALLY_UNAVAILABLE,
            "No torch wheel on this platform — MESIE numpy SpectralGPT + ChaosCUDA path is the real compute plane, not 'can't do ML'",
            step,
        )
    if any(c.name == "runtime.ghc" and c.status == CapabilityStatus.TECHNICALLY_UNAVAILABLE for c in caps):
        _add(
            caps,
            "diagnosis.haskell",
            CapabilityStatus.TECHNICALLY_UNAVAILABLE,
            "GHC not installed — bindings/haskell provides semantic AuroCompute.hs; not a total polyglot failure",
            step,
        )
    steps["7_failure_sources"] = sources

    # ------------------------------------------------------------------ 8. Report constraints only after classification
    step = 8
    n_avail = sum(
        1
        for c in caps
        if c.status in (CapabilityStatus.AVAILABLE, CapabilityStatus.AVAILABLE_BUT_UNDISCOVERED)
    )
    _add(
        caps,
        "discovery.complete",
        CapabilityStatus.AVAILABLE,
        f"{n_avail}/{len(caps)} available-class; constraints classified not collapsed",
        step,
        n_available=n_avail,
        n_total=len(caps),
    )
    steps["8_report"] = {
        "n_available_class": n_avail,
        "n_total": len(caps),
        "constraint_classes": sources,
    }

    return CapabilityReport(
        ok=True,
        repo_root=str(root),
        steps=steps,
        capabilities=caps,
        elapsed_s=time.time() - t0,
    )


def run_discovery(
    repo_root: Optional[Path] = None,
    *,
    save: bool = True,
    probes: bool = True,
) -> Dict[str, Any]:
    report = discover_capabilities(repo_root, probes=probes)
    d = report.to_dict()
    if save:
        out = Path(repo_root or _REPO) / "artifacts" / "auro-capability"
        out.mkdir(parents=True, exist_ok=True)
        (out / "DISCOVERY.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
        # markdown summary
        lines = [
            "# Capability Discovery Report",
            "",
            f"Root: `{report.repo_root}`",
            f"Elapsed: {report.elapsed_s:.2f}s",
            "",
            "## Anti-pattern",
            "",
            "> Never collapse all eight statuses into **I can't**.",
            "",
            "## By status",
            "",
        ]
        for st, names in report.by_status().items():
            if not names:
                continue
            lines.append(f"### {st} ({len(names)})")
            for n in names:
                lines.append(f"- `{n}`")
            lines.append("")
        lines.append("## Available now")
        for n in report.available():
            lines.append(f"- `{n}`")
        (out / "DISCOVERY.md").write_text("\n".join(lines), encoding="utf-8")
        d["saved"] = str(out / "DISCOVERY.json")
    return d
