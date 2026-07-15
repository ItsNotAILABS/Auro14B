/**
 * MESIE Desktop — Electron Main Process
 *
 * Launches the spectral intelligence dashboard and manages
 * the Python SDK bridge for real-time spectral operations.
 */

const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
const isDev = process.argv.includes('--dev');

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: 'MESIE — Spectral Intelligence Engine',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
    backgroundColor: '#0a0e1a',
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// ---------------------------------------------------------------------------
// Python SDK Bridge
// ---------------------------------------------------------------------------

/**
 * Run a Python script safely by passing parameters via stdin as JSON.
 * @param {string} script - Python code (must read params from stdin via json.loads(input()))
 * @param {object|null} params - Parameters to pass via stdin (null if no params needed)
 */
function runPython(script, params = null) {
  return new Promise((resolve, reject) => {
    const python = process.env.MESIE_PYTHON || 'python';
    const proc = spawn(python, ['-c', script], {
      cwd: path.resolve(__dirname, '../..'),
    });
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });
    proc.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(stdout));
        } catch {
          resolve(stdout.trim());
        }
      } else {
        reject(new Error(stderr || `Process exited with code ${code}`));
      }
    });
    if (params !== null) {
      proc.stdin.write(JSON.stringify(params));
      proc.stdin.end();
    } else {
      proc.stdin.end();
    }
  });
}

// IPC Handlers — Bridge between Renderer and Python SDK
ipcMain.handle('mesie:version', async () => {
  return runPython(`
import json, mesie
print(json.dumps({"version": mesie.__version__}))
`);
});

ipcMain.handle('mesie:validate', async (event, recordPath) => {
  return runPython(`
import json, sys
params = json.loads(sys.stdin.read())
from mesie import load_record, validate_record
rec = load_record(params["path"])
v = validate_record(rec)
print(json.dumps({"is_valid": v.is_valid, "level": v.level, "errors": v.errors, "warnings": v.warnings}))
`, { path: recordPath });
});

ipcMain.handle('mesie:generate', async (event, { type, seed }) => {
  return runPython(`
import json, sys, numpy as np
params = json.loads(sys.stdin.read())
from mesie import generate_psd, generate_fas
from mesie.core.config import GenerationConfig
cfg = GenerationConfig(seed=params["seed"])
gen_fn = generate_psd if params["type"] == "psd" else generate_fas
rec = gen_fn(config=cfg)
c = rec.components[0]
print(json.dumps({
    "record_id": rec.record_id,
    "frequency": c.frequency.tolist(),
    "amplitude": c.amplitude.tolist(),
    "n_points": len(c.frequency)
}))
`, { type: type || 'psd', seed: seed || 42 });
});

ipcMain.handle('mesie:match', async (event, { refPath, candPath }) => {
  return runPython(`
import json, sys
params = json.loads(sys.stdin.read())
from mesie import load_record, match_records
ref = load_record(params["ref"])
cand = load_record(params["cand"])
result = match_records(ref, cand)
print(json.dumps({"composite_score": result.composite_score, "metrics": result.metric_breakdown}))
`, { ref: refPath, cand: candPath });
});

ipcMain.handle('mesie:embed', async (event, recordPath) => {
  return runPython(`
import json, sys
params = json.loads(sys.stdin.read())
from mesie import load_record
from mesie.embeddings import SpectralVectorizer
rec = load_record(params["path"])
v = SpectralVectorizer()
emb = v.transform(rec)
print(json.dumps({"dimension": len(emb), "embedding": emb.tolist()}))
`, { path: recordPath });
});

ipcMain.handle('mesie:knowledge-stats', async () => {
  return runPython(`
import json
from mesie.sdk import MAESIClient
client = MAESIClient(fast=False, use_fingerprint=False)
stats = client.knowledge_stats()
print(json.dumps({
    "physical_laws": stats.physical_laws,
    "chemical_elements": stats.chemical_elements,
    "biological_systems": stats.biological_systems,
    "technical_concepts": stats.technical_concepts,
    "research_entries": stats.research_entries,
}))
`);
});

ipcMain.handle('mesie:search-research', async (event, { query, topK }) => {
  return runPython(`
import json, sys
params = json.loads(sys.stdin.read())
from mesie.sdk import search_research
hits = search_research(params["query"], top_k=params["top_k"])
print(json.dumps([{"title": h.title, "field": h.field.value if hasattr(h.field, 'value') else str(h.field)} for h in hits]))
`, { query, top_k: topK || 5 });
});

ipcMain.handle('mesie:monte-carlo', async (event, trials) => {
  return runPython(`
import json, sys, time
params = json.loads(sys.stdin.read())
sys.path.insert(0, '.')
from scripts.monte_carlo_enterprise_benchmark import MonteCarloEnterpriseRunner
runner = MonteCarloEnterpriseRunner(seed=42)
data = runner.run_all(n_trials=params["trials"])
print(json.dumps({
    "total_trials": data["total_trials"],
    "success_rate": data["overall_success_rate"],
    "enterprise_grade": data["enterprise_grade"],
    "elapsed_s": data["elapsed_s"],
    "use_cases": [{"name": uc["name"], "industry": uc["industry"], "success_rate": uc["monte_carlo"]["success_rate"]} for uc in data["use_cases"]]
}))
`, { trials: trials || 100 });
});

ipcMain.handle('dialog:open-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Spectral Records', extensions: ['json', 'csv'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });
  return result.canceled ? null : result.filePaths[0];
});
