# Host control for RO14B / Auro14B + Claude / Grok

**RO14B** = this repo (`Auro14B`) — native LMR + MESIE compute plane.  
**PARALLAX** = `E:\PARALLAX-Exchange-Clearinghouse` — sovereign DEX organism (873ms Schumann settlement heart).  
**POCKET** = host co-pilot on `:8787` — UIA, desktop, mesh, Fusion Sense.

## What you get

POCKET host powers are exposed on the **Auro API** so models and external AIs can:

| Tool | Capability |
|------|------------|
| `host_status` | POCKET runtime alive? |
| `host_open_app` | Open Edge / Code / Explorer / Notepad… |
| `host_sense_page` | Fusion Sense full-page symbols |
| `host_screenshot` | Capture screen |
| `host_click` | Click by coords / query |
| `host_dispatch` | Mesh `@ARCHON` / `@DESIGN` / headless |
| `host_ms_protocol` | Microsoft host protocol bridge |

## Run (two processes)

```powershell
# 1) POCKET host (leave running)
cd C:\Users\Medin\OneDrive\pocket-os
$env:PYTHONPATH="C:\Users\Medin\OneDrive\pocket-os\src"
$env:POCKET_MESH_HOOK="0"
python -m pocket serve --host 127.0.0.1 --port 8787

# 2) Auro API (RO14B)
cd C:\Users\Medin\Documents\GitHub\Auro14B
$env:PYTHONPATH="."
$env:POCKET_URL="http://127.0.0.1:8787"
python -m auro_native_llm.server.app --host 127.0.0.1 --port 8765
```

## Endpoints (Auro :8765)

```http
GET  /v1/tools
GET  /v1/host/status
POST /v1/host/invoke   {"name":"host_open_app","arguments":{"app":"notepad"}}
POST /v1/host/sense    {"prompt":"full page"}
POST /v1/host/dispatch {"agent":"DESIGN","message":"polish desk"}
POST /v1/host/ms       {"action":"status"}
```

## For Claude / Grok / Codex

1. `GET http://127.0.0.1:8765/v1/tools` → tool definitions  
2. Model emits tool call → `POST /v1/host/invoke`  
3. Feed result back into the model  

OpenAPI-style names match OpenAI function tools.

## Parallax link

PARALLAX settlement cycle is **873ms** (same constant as POCKET `runtime-worker` heart).  
Exchange clearinghouse is separate from host co-pilot; RO14B can call host tools while Parallax handles ICP/DEX organism logic.

## Models inside Auro

Any Auro mind / HIM / colony path can call `auro_native_llm.host_control.invoke(...)` from Python, or hit HTTP tools above. Same API for external AIs.
