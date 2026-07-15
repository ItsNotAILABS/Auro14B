/**
 * MESIE Desktop — Renderer Process
 *
 * Handles UI interactions and communicates with the Python SDK
 * via the preload-exposed `window.mesie` bridge.
 */

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

document.querySelectorAll('.nav-item').forEach((item) => {
  item.addEventListener('click', () => {
    // Update active nav
    document.querySelectorAll('.nav-item').forEach((n) => n.classList.remove('active'));
    item.classList.add('active');

    // Show panel
    const panelId = `panel-${item.dataset.panel}`;
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    document.getElementById(panelId).classList.add('active');
  });
});

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

async function loadDashboard() {
  try {
    const ver = await window.mesie.getVersion();
    document.getElementById('stat-version').textContent = ver.version;
    document.getElementById('version-badge').textContent = `v${ver.version}`;

    const stats = await window.mesie.knowledgeStats();
    document.getElementById('stat-laws').textContent = stats.physical_laws;
    document.getElementById('stat-elements').textContent = stats.chemical_elements;
    document.getElementById('stat-research').textContent = stats.research_entries;

    setStatus('Engine ready', 'green');
  } catch (err) {
    setStatus(`Error: ${err.message}`, 'red');
  }
}

function setStatus(text, color) {
  document.getElementById('status-text').textContent = text;
  const dot = document.querySelector('.status-dot');
  dot.className = `status-dot ${color}`;
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

document.getElementById('btn-generate').addEventListener('click', async () => {
  const type = document.getElementById('gen-type').value;
  const seed = parseInt(document.getElementById('gen-seed').value, 10);
  const resultEl = document.getElementById('gen-result');

  resultEl.innerHTML = '<span class="highlight">Generating...</span>';
  try {
    const data = await window.mesie.generate({ type, seed });
    resultEl.innerHTML = `<span class="success">✓ Generated ${escapeHtml(type.toUpperCase())}</span>\n` +
      `Record ID: ${escapeHtml(String(data.record_id))}\n` +
      `Points: ${escapeHtml(String(data.n_points))}\n` +
      `Freq range: [${escapeHtml(data.frequency[0].toFixed(2))}, ${escapeHtml(data.frequency[data.frequency.length - 1].toFixed(2))}] Hz`;

    drawSpectrum(data.frequency, data.amplitude);
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  }
});

function drawSpectrum(freq, amp) {
  const canvas = document.getElementById('spectrum-canvas');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = '#1a2235';
  ctx.fillRect(0, 0, w, h);

  // Grid
  ctx.strokeStyle = '#2a3a5a';
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 5; i++) {
    const y = (h / 5) * (i + 1);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Spectrum line
  const maxAmp = Math.max(...amp);
  const minFreq = freq[0];
  const maxFreq = freq[freq.length - 1];

  ctx.beginPath();
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;

  for (let i = 0; i < freq.length; i++) {
    const x = ((freq[i] - minFreq) / (maxFreq - minFreq)) * w;
    const y = h - (amp[i] / maxAmp) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill under curve
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
  ctx.fill();

  // Labels
  ctx.fillStyle = '#64748b';
  ctx.font = '11px monospace';
  ctx.fillText(`${minFreq.toFixed(1)} Hz`, 5, h - 5);
  ctx.fillText(`${maxFreq.toFixed(1)} Hz`, w - 60, h - 5);
  ctx.fillText(`Max: ${maxAmp.toFixed(4)}`, w - 120, 15);
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

document.getElementById('btn-validate-browse').addEventListener('click', async () => {
  const path = await window.mesie.openFile();
  if (path) document.getElementById('validate-path').value = path;
});

document.getElementById('btn-validate').addEventListener('click', async () => {
  const path = document.getElementById('validate-path').value;
  const resultEl = document.getElementById('validate-result');
  if (!path) { resultEl.innerHTML = '<span class="error">Please select a file</span>'; return; }

  resultEl.innerHTML = '<span class="highlight">Validating...</span>';
  try {
    const data = await window.mesie.validate(path);
    const icon = data.is_valid ? '✓' : '✗';
    const cls = data.is_valid ? 'success' : 'error';
    resultEl.innerHTML = `<span class="${cls}">${icon} ${data.is_valid ? 'VALID' : 'INVALID'}</span>\n` +
      `Level: ${escapeHtml(String(data.level))}/6\n` +
      (data.errors.length ? `Errors: ${escapeHtml(data.errors.join(', '))}\n` : '') +
      (data.warnings.length ? `Warnings: ${escapeHtml(data.warnings.join(', '))}` : '');
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  }
});

// ---------------------------------------------------------------------------
// Match
// ---------------------------------------------------------------------------

document.getElementById('btn-match-ref-browse').addEventListener('click', async () => {
  const path = await window.mesie.openFile();
  if (path) document.getElementById('match-ref-path').value = path;
});

document.getElementById('btn-match-cand-browse').addEventListener('click', async () => {
  const path = await window.mesie.openFile();
  if (path) document.getElementById('match-cand-path').value = path;
});

document.getElementById('btn-match').addEventListener('click', async () => {
  const refPath = document.getElementById('match-ref-path').value;
  const candPath = document.getElementById('match-cand-path').value;
  const resultEl = document.getElementById('match-result');
  if (!refPath || !candPath) { resultEl.innerHTML = '<span class="error">Please select both files</span>'; return; }

  resultEl.innerHTML = '<span class="highlight">Matching...</span>';
  try {
    const data = await window.mesie.match({ refPath, candPath });
    const score = (data.composite_score * 100).toFixed(1);
    resultEl.innerHTML = `<span class="success">Composite Score: ${escapeHtml(score)}%</span>\n\n` +
      'Metric Breakdown:\n' +
      Object.entries(data.metrics).map(([k, v]) => `  ${escapeHtml(k)}: ${(v * 100).toFixed(1)}%`).join('\n');
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  }
});

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

document.getElementById('btn-embed-browse').addEventListener('click', async () => {
  const path = await window.mesie.openFile();
  if (path) document.getElementById('embed-path').value = path;
});

document.getElementById('btn-embed').addEventListener('click', async () => {
  const path = document.getElementById('embed-path').value;
  const resultEl = document.getElementById('embed-result');
  if (!path) { resultEl.innerHTML = '<span class="error">Please select a file</span>'; return; }

  resultEl.innerHTML = '<span class="highlight">Computing embedding...</span>';
  try {
    const data = await window.mesie.embed(path);
    resultEl.innerHTML = `<span class="success">✓ Embedding computed</span>\n` +
      `Dimension: ${data.dimension}\n` +
      `Vector: [${data.embedding.slice(0, 8).map(v => v.toFixed(4)).join(', ')}${data.dimension > 8 ? ', ...' : ''}]`;
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  }
});

// ---------------------------------------------------------------------------
// Knowledge
// ---------------------------------------------------------------------------

document.getElementById('btn-search').addEventListener('click', async () => {
  const query = document.getElementById('knowledge-query').value;
  const resultEl = document.getElementById('knowledge-result');
  if (!query) { resultEl.innerHTML = '<span class="error">Enter a search query</span>'; return; }

  resultEl.innerHTML = '<span class="highlight">Searching...</span>';
  try {
    const hits = await window.mesie.searchResearch({ query, topK: 10 });
    if (hits.length === 0) {
      resultEl.innerHTML = '<span class="error">No results found</span>';
    } else {
      resultEl.innerHTML = `<span class="success">Found ${hits.length} results:</span>\n\n` +
        hits.map((h, i) => `${i + 1}. [${escapeHtml(h.field)}] ${escapeHtml(h.title)}`).join('\n');
    }
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  }
});

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

document.getElementById('btn-benchmark').addEventListener('click', async () => {
  const trials = parseInt(document.getElementById('mc-trials').value, 10);
  const resultEl = document.getElementById('benchmark-result');
  const btn = document.getElementById('btn-benchmark');

  btn.disabled = true;
  resultEl.innerHTML = `<span class="highlight">Running Monte Carlo benchmark (${trials} × 10 = ${trials * 10} trials)...</span>\n\nThis may take a few seconds.`;

  try {
    const data = await window.mesie.monteCarlo(trials);
    const grade = data.enterprise_grade ? '✓ PASS' : '✗ REVIEW';
    const gradeCls = data.enterprise_grade ? 'success' : 'warning';

    resultEl.innerHTML = `<span class="${gradeCls}">Enterprise Grade: ${grade}</span>\n` +
      `Total Trials: ${data.total_trials}\n` +
      `Overall Success: ${(data.success_rate * 100).toFixed(1)}%\n` +
      `Runtime: ${data.elapsed_s}s\n\n` +
      'Use Cases:\n' +
      data.use_cases.map((uc) =>
        `  [${escapeHtml(uc.industry)}] ${escapeHtml(uc.name)}: ${(uc.success_rate * 100).toFixed(1)}%`
      ).join('\n');
  } catch (err) {
    resultEl.innerHTML = `<span class="error">✗ ${escapeHtml(err.message)}</span>`;
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', loadDashboard);
