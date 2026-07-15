# MESIE Desktop

Electron-based desktop application for the MESIE Spectral Intelligence Engine.

## Features

- **Dashboard** — Real-time knowledge base stats and engine status
- **Generate** — Create PSD/FAS spectra with interactive visualization
- **Validate** — Validate spectral record files with detailed reports
- **Match** — Compare two spectral records with composite scoring
- **Embed** — Compute spectral vector embeddings
- **Knowledge** — Search the MAESI research catalog
- **Benchmark** — Run Monte Carlo enterprise benchmarks with live results

## Requirements

- Node.js 18+
- Python 3.10+ with MESIE installed (`pip install -e ".[full]"`)
- Electron 28+

## Quick Start

```bash
# Install dependencies
cd mesie-desktop
npm install

# Run in development mode
npm run dev

# Run in production mode
npm start
```

## Build Distributable

```bash
# Windows
npm run build:win

# macOS
npm run build:mac

# Linux
npm run build:linux
```

## PowerShell Launch

```powershell
Import-Module ../scripts/MESIE.psm1
Start-MESIEDesktop -Dev
```

## Architecture

```
mesie-desktop/
├── src/
│   ├── main.js        # Electron main process + Python SDK bridge
│   ├── preload.js     # Context bridge (secure IPC)
│   ├── index.html     # Dashboard UI
│   ├── styles.css     # Dark spectral theme
│   └── renderer.js    # UI logic and event handling
└── package.json       # Electron + builder config
```

The app communicates with the Python MESIE SDK via child process spawning in the main process. All SDK calls go through the secure IPC bridge defined in `preload.js`.
