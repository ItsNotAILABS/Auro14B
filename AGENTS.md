# AGENTS.md

## Cursor Cloud specific instructions

MESIE is a polyglot monorepo. The **primary product is the Python engine** (`mesie/`,
`spectral_engine.py`), exercised via `pytest`, the `mesie` CLI, and the `examples/` scripts.
Two runnable pieces were set up and verified in this environment:

### 1. Python engine (core product)
- Deps are installed by the update script (`pip install -e ".[dev,full]"`), matching CI (`.github/workflows/ci.yml`).
- This image only has `python3`/`pip` on PATH — **there is no `python` alias**. Use `python3 -m pytest`, `python3 examples/...`, etc.
- Console scripts (`mesie`, `mesie-tools`) install to `~/.local/bin`, which is **not on PATH** by default. Either add it (`export PATH="$HOME/.local/bin:$PATH"`) or invoke modules directly (`python3 -m mesie.cli ...`).
- Run tests: `python3 -m pytest` (see `pyproject.toml` `testpaths=["tests"]`). Coverage variant: `python3 -m pytest --cov=mesie`.
- No linter/formatter is configured in the repo (no ruff/flake8/black config); CONTRIBUTING.md only asks for PEP 8 by hand. "Lint" effectively means `pytest`.
- Known pre-existing failure (not an environment issue): `tests/test_sdk_and_corpus.py::TestSDK::test_version` asserts the old `0.3.0` version while `pyproject.toml` is `0.4.0`. Everything else passes (2209 passed, 3 skipped).
- CLI/`load_record` expect the simple record schema (`components[].frequency` / `.amplitude`), e.g. `examples/02_match_two_records.py`. The domain files under `data/spectral_library/` and `data/reference/` use richer, non-record schemas and are **not** valid inputs to `mesie info`.
- The `[ml]`/`[ai]`/`[intelligence]` extras (torch/transformers) are heavy and optional; the core matching/generation/validation/CLI work without them.

### 2. Cloudflare Worker API (`workers/mesie-api/`)
- Standalone TypeScript Worker; does **not** import the Python package.
- Install with `npm install` in `workers/mesie-api/` (only this subdir and `mesie-desktop/` have Node projects; only this one has a lockfile).
- Run dev server: `npx wrangler dev` (bind explicitly for headless: `npx wrangler dev --port 8787 --ip 127.0.0.1`). Ready at `http://127.0.0.1:8787`.
- Endpoints: `GET /health`, `GET /v1/datasets`, `POST /v1/validate`, `POST /v1/match`. Auth is open locally (`MESIE_API_KEY`/`MESIE_PUBLIC` unset). Run it in a persistent tmux session, not a one-shot background command.

### Other bindings (not set up here; optional)
- `mesie-desktop/` (Electron) needs a display (use `xvfb-run`) plus a `mesie`-importable Python interpreter; spawns Python from repo root.
- `bindings/` (Rust/Cargo, Julia, Zig, Motoko) each need their own toolchains and are optional polyglot ports of the engine.
