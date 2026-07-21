"""Launch the native Auro/HIM runtime and Cloudflare commercial UI together.

No Workers AI binding and no external model fallback are used. The launcher
starts the repository-native production API, waits for readiness, builds and
starts the Cloudflare Worker locally, verifies the gateway, opens the browser,
and shuts both processes down together.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "auro14b-app"


def command(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    if os.name == "nt":
        found = shutil.which(f"{name}.cmd")
        if found:
            return found
    raise SystemExit(f"Required command not found: {name}")


def wait_for(url: str, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = str(error)
        time.sleep(0.4)
    raise RuntimeError(f"{label} did not become ready: {last_error}")


def stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Use Auro/HIM through its commercial Cloudflare interface")
    parser.add_argument("--checkpoint", default="checkpoints/open/HIM-native-v0")
    parser.add_argument("--native-port", type=int, default=8090)
    parser.add_argument("--ui-port", type=int, default=8787)
    parser.add_argument("--context-tokens", type=int, default=32000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate prerequisites without starting services")
    args = parser.parse_args()

    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Native checkpoint not found: {checkpoint}")
    if not APP.exists():
        raise SystemExit(f"Commercial application not found: {APP}")

    python = sys.executable
    npm = command("npm")
    npx = command("npx")

    if args.check:
        print("Auro launch prerequisites verified")
        print(f"Checkpoint: {checkpoint}")
        print(f"Python: {python}")
        print(f"npm: {npm}")
        print(f"npx: {npx}")
        return 0

    native_url = f"http://127.0.0.1:{args.native_port}"
    ui_url = f"http://127.0.0.1:{args.ui_port}"
    native: subprocess.Popen[bytes] | None = None
    worker: subprocess.Popen[bytes] | None = None

    try:
        native = subprocess.Popen(
            [
                python,
                str(ROOT / "scripts" / "launch_him.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.native_port),
                "--checkpoint",
                str(checkpoint),
                "--context-tokens",
                str(args.context_tokens),
                "--no-browser",
            ],
            cwd=ROOT,
        )
        wait_for(f"{native_url}/v1/health/ready", 45, "native Auro/HIM runtime")

        if not args.skip_install and not (APP / "node_modules").exists():
            subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=APP, check=True)
        subprocess.run([npm, "run", "build"], cwd=APP, check=True)

        worker = subprocess.Popen(
            [
                npx,
                "wrangler",
                "dev",
                "--local",
                "--port",
                str(args.ui_port),
                "--var",
                f"AURO_NATIVE_BASE_URL:{native_url}",
                "--var",
                "AURO_NATIVE_MODEL:auro-him",
            ],
            cwd=APP,
        )
        wait_for(f"{ui_url}/api/health", 60, "commercial Cloudflare interface")

        print("\nAURO IS READY")
        print(f"Commercial interface: {ui_url}")
        print(f"Native runtime: {native_url}")
        print(f"Checkpoint: {checkpoint}")
        print("Press Ctrl+C to stop both services.\n")
        if not args.no_browser:
            webbrowser.open(ui_url)

        while native.poll() is None and worker.poll() is None:
            time.sleep(0.5)
        if native.poll() is not None:
            raise RuntimeError(f"native runtime exited with code {native.returncode}")
        raise RuntimeError(f"commercial interface exited with code {worker.returncode}")
    except KeyboardInterrupt:
        return 0
    finally:
        stop(worker)
        stop(native)


if __name__ == "__main__":
    raise SystemExit(main())
